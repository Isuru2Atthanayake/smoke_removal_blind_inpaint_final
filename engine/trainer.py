import os
import torch
import wandb
import glog as log
# import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from modeling.architecture import MPN, RIN, Discriminator, PatchDiscriminator
from losses.bce import WeightedBCELoss
from losses.consistency import SemanticConsistencyLoss, IDMRFLoss
from losses.adversarial import compute_gradient_penalty
from utils.mask_utils import MaskGenerator, ConfidenceDrivenMaskLayer, COLORS
from utils.data_utils import linear_scaling, linear_unscaling, get_random_string, RaindropDataset

# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, opt):
        print("we are at trainer")
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() in ["smoke_dataset", "places"]
        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.JOINT.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)

        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME)

        self.transform = transforms.Compose([transforms.Resize(self.opt.DATASET.SIZE),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                             ])


        self.dataset = ImageFolder(root=self.opt.DATASET.ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)

        self.imagenet_transform = transforms.Compose([transforms.RandomCrop(self.opt.DATASET.SIZE, pad_if_needed=True, padding_mode="reflect"),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                                      ])
        if self.opt.DATASET.NAME.lower() == "smoke_dataset":
            celeb_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.transform)
            imagenet_dataset = ImageFolder(root=self.opt.DATASET.IMAGENET, transform=self.imagenet_transform)
            self.cont_dataset = torch.utils.data.ConcatDataset([celeb_dataset, imagenet_dataset])
        else:
            self.cont_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.imagenet_transform)
        self.cont_image_loader = data.DataLoader(dataset=self.cont_dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.mask_generator = MaskGenerator(self.opt.MASK)
        self.mask_smoother = ConfidenceDrivenMaskLayer(self.opt.MASK.GAUS_K_SIZE, self.opt.MASK.SIGMA)
        # self.mask_smoother = GaussianSmoothing(1, 5, 1/40)

        self.to_pil = transforms.ToPILImage()

        self.mpn = MPN(base_n_channels=self.opt.MODEL.MPN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.rin = RIN(base_n_channels=self.opt.MODEL.RIN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.discriminator = Discriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)
        self.patch_discriminator = PatchDiscriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)

        self.optimizer_mpn = torch.optim.Adam(self.mpn.parameters(), lr=self.opt.MODEL.MPN.LR, betas=self.opt.MODEL.MPN.BETAS)
        self.optimizer_rin = torch.optim.Adam(self.rin.parameters(), lr=self.opt.MODEL.RIN.LR, betas=self.opt.MODEL.RIN.BETAS)
        self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.patch_discriminator.parameters()), lr=self.opt.MODEL.D.LR, betas=self.opt.MODEL.D.BETAS)
        self.optimizer_joint = torch.optim.Adam(list(self.mpn.parameters()) + list(self.rin.parameters()), lr=self.opt.MODEL.JOINT.LR, betas=self.opt.MODEL.JOINT.BETAS)

        self.num_step = self.opt.TRAIN.START_STEP

        if self.opt.TRAIN.START_STEP != 0 and self.opt.TRAIN.RESUME:  # find start step from checkpoint file name. TODO
            log.info("Checkpoints loading...")
            self.load_checkpoints(self.opt.TRAIN.START_STEP)

        self.check_and_use_multi_gpu()

        self.weighted_bce_loss = WeightedBCELoss().cuda()
        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()

        # adding the check_data_input function to check the input data
    def check_data_input(self, imgs, y_imgs, masks):
            # Print out statistics
            print("Statistics of thee imgs:")
            print("Mean:", torch.mean(imgs))
            print("Standard Deviation:", torch.std(imgs))

            # Check for NaN values
            if torch.isnan(imgs).any():
                print("imgs tensor contains NaN values.")
            else:
                print("imgs tensor does not contain NaN values.")

            # Check for infinite values
            if torch.isinf(imgs).any():
                print("imgs tensor contains infinite values.")
            else:
                print("imgs tensor does not contain infinite values.")

    # to plot the images
    def run(self):
        while self.num_step < self.opt.TRAIN.NUM_TOTAL_STEP:
            self.num_step += 1
            info = " [Step: {}/{} ({}%)] ".format(self.num_step, self.opt.TRAIN.NUM_TOTAL_STEP, 100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP)

            imgs, _ = next(iter(self.image_loader))
            y_imgs = imgs.float().cuda()
            imgs = linear_scaling(imgs.float().cuda())
            batch_size, channels, h, w = imgs.size()

            masks = torch.from_numpy(self.mask_generator.generate(h, w)).repeat([batch_size, 1, 1, 1]).float().cuda()

            cont_imgs, _ = next(iter(self.cont_image_loader))
            cont_imgs = linear_scaling(cont_imgs.float().cuda())
            if cont_imgs.size(0) != imgs.size(0):
                cont_imgs = cont_imgs[:imgs.size(0)]

            smooth_masks = self.mask_smoother(1 - masks) + masks
            smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

            masked_imgs = cont_imgs * smooth_masks + imgs * (1. - smooth_masks)
            self.unknown_pixel_ratio = torch.sum(masks.view(batch_size, -1), dim=1).mean() / (h * w)

            # Call the check_data_input function here
            self.check_data_input(imgs, y_imgs, masks)

            # print("Training D...")
            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                d_loss = self.train_D(masked_imgs, masks, y_imgs)
            info += "D Loss: {} ".format(d_loss)

            # print("Training G...")
            m_loss, g_loss, pred_masks, output = self.train_G(masked_imgs, masks, y_imgs)
            info += "M Loss: {} G Loss: {} ".format(m_loss, g_loss)

            # print("Logging and visualization...")
            if self.num_step % self.opt.TRAIN.LOG_INTERVAL == 0:
                log.info(info)

            if self.num_step % self.opt.TRAIN.VISUALIZE_INTERVAL == 0:
                idx = self.opt.WANDB.NUM_ROW
                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(cont_imgs[idx]).cpu()), caption="contaminant_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(masked_imgs[idx]).cpu()), caption="masked_image"),
                    self.wandb.Image(self.to_pil(masks[idx].cpu()), caption="original_masks"),
                    self.wandb.Image(self.to_pil(smooth_masks[idx].cpu()), caption="smoothed_masks"),
                    self.wandb.Image(self.to_pil(pred_masks[idx].cpu()), caption="predicted_masks"),
                    self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                ]}, commit=False)
            self.wandb.log({})
            if self.num_step % self.opt.TRAIN.SAVE_INTERVAL == 0 and self.num_step != 0:
                self.do_checkpoint(self.num_step)

    # print("Training loop ended.")

# To check the input data
    # def check_data_input(self, imgs, y_imgs, masks):
    #     # Print out statistics
    #     print("Statistics of imgs:")
    #     print("Mean:", torch.mean(imgs))
    #     print("Standard Deviation:", torch.std(imgs))
    #
    #     # Check for NaN values
    #     if torch.isnan(imgs).any():
    #         print("imgs tensor contains NaN values.")
    #     else:
    #         print("imgs tensor does not contain NaN values.")
    #
    #     # Check for infinite values
    #     if torch.isinf(imgs).any():
    #         print("imgs tensor contains infinite values.")
    #     else:
    #         print("imgs tensor does not contain infinite values.")

    def train_D(self, x, y_masks, y):
        print("train_D started")
        self.optimizer_discriminator.zero_grad()

        pred_masks, neck = self.mpn(x)
        output = self.rin(x, pred_masks, neck)

        real_global_validity = self.discriminator(y).mean()
        fake_global_validity = self.discriminator(output.detach()).mean()
        gp_global = compute_gradient_penalty(self.discriminator, output.data, y.data)

        real_patch_validity = self.patch_discriminator(y, y_masks).mean()
        fake_patch_validity = self.patch_discriminator(output.detach(), y_masks).mean()
        gp_fake = compute_gradient_penalty(self.patch_discriminator, output.data, y.data, y_masks)

        real_validity = real_global_validity + real_patch_validity
        fake_validity = fake_global_validity + fake_patch_validity
        gp = gp_global + gp_fake

        d_loss = -real_validity + fake_validity + self.opt.OPTIM.GP * gp
        d_loss.backward()
        self.optimizer_discriminator.step()

        self.wandb.log({"real_global_validity": -real_global_validity.item(),
                        "fake_global_validity": fake_global_validity.item(),
                        "real_patch_validity": -real_patch_validity.item(),
                        "fake_patch_validity": fake_patch_validity.item(),
                        "gp_global": gp_global.item(),
                        "gp_fake": gp_fake.item(),
                        "real_validity": -real_validity.item(),
                        "fake_validity": fake_validity.item(),
                        "gp": gp.item()}, commit=False)
        return d_loss.item()
        print("train_D ended ")
    def train_G(self, x, y_masks, y):
        print("train_G started ")
        if self.num_step < self.opt.TRAIN.NUM_STEPS_FOR_JOINT:
            self.optimizer_mpn.zero_grad()
            self.optimizer_rin.zero_grad()

            pred_masks, neck = self.mpn(x)
            m_loss = self.weighted_bce_loss(pred_masks, y_masks, torch.tensor([1 - self.unknown_pixel_ratio, self.unknown_pixel_ratio]))
            self.wandb.log({"m_loss/Mask Loss": m_loss.item()}, commit=False)
            m_loss = self.opt.OPTIM.MASK * m_loss
            m_loss.backward(retain_graph=True)
            self.optimizer_mpn.step()
            if self.opt.MODEL.RIN.EMBRACE:
                x_embraced = x.detach() * (1 - pred_masks.detach())
                output = self.rin(x_embraced, pred_masks.detach(), neck.detach())
            else:
                output = self.rin(x, pred_masks.detach(), neck.detach())
            recon_loss = self.reconstruction_loss(output, y)
            sem_const_loss = self.semantic_consistency_loss(output, y)
            # tex_const_loss = self.texture_consistency_loss(output, y)
            # tex_const_loss = 0
            adv_global_loss = -self.discriminator(output).mean()
            adv_patch_loss = -self.patch_discriminator(output, y_masks).mean()
            adv_loss = adv_global_loss + adv_patch_loss

            # g_loss = self.opt.OPTIM.RECON * recon_loss + \
            #          self.opt.OPTIM.SEMANTIC * sem_const_loss + \
            #          self.opt.OPTIM.TEXTURE * tex_const_loss * \
            #          self.opt.OPTIM.ADVERSARIAL * adv_loss

            g_loss = self.opt.OPTIM.RECON * recon_loss + \
                     self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                     self.opt.OPTIM.ADVERSARIAL * adv_loss

            print("The Reconstruction Loss:", recon_loss.item())  # Print reconstruction loss
            print("The Semantic Consistency Loss:", sem_const_loss.item())  # Print semantic consistency loss
            # print("The Texture Consistency Loss:", tex_const_loss.item())  # Print texture consistency loss
            print("The Adversarial Loss:", adv_loss.item())  # Print adversarial loss
            print("The G Loss:", g_loss.item())  # Print generator loss

            g_loss.backward()
            self.optimizer_rin.step()
        else:
            self.optimizer_joint.zero_grad()
            pred_masks, neck = self.mpn(x)
            m_loss = self.weighted_bce_loss(pred_masks, y_masks, torch.tensor([1 - self.unknown_pixel_ratio, self.unknown_pixel_ratio]))
            self.wandb.log({"m_loss": m_loss.item()}, commit=False)
            m_loss = self.opt.OPTIM.MASK * m_loss
            if self.opt.MODEL.RIN.EMBRACE:
                x_embraced = x.detach() * (1 - pred_masks.detach())
                output = self.rin(x_embraced, pred_masks.detach(), neck.detach())
            else:
                output = self.rin(x, pred_masks.detach(), neck.detach())
            recon_loss = self.reconstruction_loss(output, y)
            sem_const_loss = self.semantic_consistency_loss(output, y)
            # tex_const_loss = self.texture_consistency_loss(output, y)
            # tex_const_loss = 0
            adv_global_loss = -self.discriminator(output).mean()
            adv_patch_loss = -self.patch_discriminator(output, y_masks).mean()
            adv_loss = adv_global_loss + adv_patch_loss

            # g_loss = self.opt.OPTIM.RECON * recon_loss + \
            #          self.opt.OPTIM.SEMANTIC * sem_const_loss + \
            #          self.opt.OPTIM.TEXTURE * tex_const_loss + \
            #          self.opt.OPTIM.ADVERSARIAL * adv_loss

            g_loss = self.opt.OPTIM.RECON * recon_loss + \
                     self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                     self.opt.OPTIM.ADVERSARIAL * adv_loss

            final_loss = self.opt.MODEL.MPN.LOSS_COEFF * m_loss + self.opt.MODEL.RIN.LOSS_COEFF * g_loss
            final_loss.backward()
            self.optimizer_joint.step()
        self.wandb.log({"recon_loss": recon_loss.item(),
                        "sem_const_loss": sem_const_loss.item(),
                        # "tex_const_loss": tex_const_loss.item(),
                        "adv_global_loss": adv_global_loss.item(),
                        "adv_patch_loss": adv_patch_loss.item(),
                        "adv_loss": adv_loss.item()}, commit=False)
        return m_loss.item(), g_loss.item(), pred_masks.detach(), output.detach()
        print("train_G ended ")
    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.mpn = torch.nn.DataParallel(self.mpn).cuda()
            self.rin = torch.nn.DataParallel(self.rin).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            self.patch_discriminator = torch.nn.DataParallel(self.patch_discriminator).cuda()
            self.mask_smoother = torch.nn.DataParallel(self.mask_smoother).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.mpn = self.mpn.cuda()
            self.rin = self.rin.cuda()
            self.discriminator = self.discriminator.cuda()
            self.patch_discriminator = self.patch_discriminator.cuda()
            self.mask_smoother = self.mask_smoother.cuda()

    def do_checkpoint(self, num_step):
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

        checkpoint = {
            'num_step': num_step,
            'mpn': self.mpn.state_dict(),
            'rin': self.rin.state_dict(),
            'D': self.discriminator.state_dict(),
            'patch_D': self.patch_discriminator.state_dict(),
            'optimizer_mpn': self.optimizer_mpn.state_dict(),
            'optimizer_rin': self.optimizer_rin.state_dict(),
            'optimizer_joint': self.optimizer_joint.state_dict(),
            'optimizer_D': self.optimizer_discriminator.state_dict(),
            # 'scheduler_mpn': self.scheduler_mpn.state_dict(),
            # 'scheduler_rin': self.scheduler_rin.state_dict(),
            # 'scheduler_joint': self.scheduler_joint.state_dict(),
            # 'scheduler_D': self.scheduler_discriminator.state_dict(),
        }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))

    def load_checkpoints(self, num_step):
        checkpoints = torch.load("./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))
        self.num_step = checkpoints["num_step"]
        self.mpn.load_state_dict(checkpoints["mpn"])
        self.rin.load_state_dict(checkpoints["rin"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizer_mpn.load_state_dict(checkpoints["optimizer_mpn"])
        self.optimizer_rin.load_state_dict(checkpoints["optimizer_rin"])
        self.optimizer_discriminator.load_state_dict(checkpoints["optimizer_D"])
        self.optimizer_joint.load_state_dict(checkpoints["optimizer_joint"])
        self.optimizers_to_cuda()

        # self.scheduler_mpn.load_state_dict(checkpoints["scheduler_mpn"])
        # self.scheduler_rin.load_state_dict(checkpoints["scheduler_rin"])
        # self.scheduler_discriminator.load_state_dict(checkpoints["scheduler_D"])
        # self.scheduler_joint.load_state_dict(checkpoints["scheduler_joint"])

    def optimizers_to_cuda(self):
        for state in self.optimizer_mpn.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_rin.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_joint.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()


class RaindropTrainer(Trainer):
    def __init__(self, opt):
        print("RaindropTrainer class started")
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() == "smoke_dataset"

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.JOINT.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)

        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME)

        self.transform = transforms.Compose([transforms.Resize(self.opt.DATASET.SIZE),
                                             transforms.CenterCrop(self.opt.DATASET.SIZE),
                                             transforms.ToTensor()
                                             ])
        # Modify this line to include your dataset folder
        # self.train_dataset = ImageFolder(root=os.path.join(self.opt.DATASET.ROOT, 'train'),
        #                                  transform=self.transform,
        #                                  target_transform=lambda x: 0 if 'mask' in x else (1 if 'smoke' in x else 2))
        #
        # self.train_dataset = ImageFolder(root=os.path.join(self.opt.DATASET.ROOT, 'train'),
        #                                  transform=self.transform,
        #                                  target_transform=lambda x: 0 if 'mask' in x[0] else (
        #                                      1 if 'smoke' in x[0] else 2))
        #
        # for i, (image, label) in enumerate(self.train_dataset):
        #     print(f"Loaded image file: {self.train_dataset.imgs[i][0]}")

        self.dataset = RaindropDataset(root=self.opt.DATASET.RAINDROP_ROOT, transform=self.transform, target_transform=self.transform)
        # To ckeck the dataset
        if len(self.dataset) == 0:
            print("No dataset loaded.")
        else:
            print("Dataset loaded Dataset size:", len(self.dataset))

        print("a")
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)
        print("b")
        self.to_pil = transforms.ToPILImage()
        print("c")
        print("Initializing MPN, RIN, and Discriminator...")
        self.mpn = MPN(base_n_channels=self.opt.MODEL.MPN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.rin = RIN(base_n_channels=self.opt.MODEL.RIN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.discriminator = Discriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)
        print("Models initialized.")
        print("Initializing optimizers...")
        self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()), lr=self.opt.MODEL.D.LR, betas=self.opt.MODEL.D.BETAS)
        self.optimizer_joint = torch.optim.Adam(list(self.mpn.parameters()) + list(self.rin.parameters()), lr=self.opt.MODEL.JOINT.LR, betas=self.opt.MODEL.JOINT.BETAS)
        print("Optimizers initialized.")

        self.num_step = self.opt.TRAIN.START_STEP ## this is the step from where we want to start the training
        print("g")
        log.info("Checkpoints loading...")
        self.load_checkpoints(self.opt.TRAIN.START_STEP)
        print("h")
        self.check_and_use_multi_gpu()
        print("i")
        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()
        print("RaindropTrainer class end")
    def run(self):
        print("Starting the run method of RaindropTrainer...")
        while self.num_step < self.opt.TRAIN.NUM_TOTAL_STEP:
            self.num_step += 1
            info = " [Step: {}/{} ({}%)] ".format(self.num_step, self.opt.TRAIN.NUM_TOTAL_STEP, 100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP)

            imgs, y_imgs = next(iter(self.image_loader))
            print("Batch loaded. Image shape:", imgs.shape, "Label shape:", y_imgs.shape)
            imgs = linear_scaling(imgs.float().cuda())
            y_imgs = y_imgs.float().cuda()


            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                print("for _ in range(self.opt.MODEL.D.NUM_CRITICS): started")
                self.optimizer_discriminator.zero_grad()

               # ganing smoke freee image
                pred_masks, neck = self.mpn(imgs)
                output = self.rin(imgs, pred_masks, neck)

                real_validity = self.discriminator(y_imgs).mean()
                fake_validity = self.discriminator(output.detach()).mean()
                gp = compute_gradient_penalty(self.discriminator, output.data, y_imgs.data)

                d_loss = -real_validity + fake_validity + self.opt.OPTIM.GP * gp
                d_loss.backward()
                self.optimizer_discriminator.step()

                self.wandb.log({"real_validity": -real_validity.item(),
                                "fake_validity": fake_validity.item(),
                                "gp": gp.item()}, commit=False)

            self.optimizer_joint.zero_grad()
            pred_masks, neck = self.mpn(imgs)
            print("pred_masks shape:", pred_masks.shape)

            if self.opt.MODEL.RIN.EMBRACE:
                print("embracing the model")
                x_embraced = imgs.detach() * (1 - pred_masks.detach())
                output = self.rin(x_embraced, pred_masks.detach(), neck.detach())
            else:
                output = self.rin(imgs, pred_masks.detach(), neck.detach())

            # Add the following line to print the raw predicted masks
            # print("Predicted Masks (raw):", pred_masks[idx])
            #
            # # Plot the raw predicted masks
            # plt.imshow(pred_masks[idx].cpu().squeeze(), cmap='gray')
            # plt.title("Predicted Masks (raw)")
            # plt.colorbar()
            # plt.show()

            recon_loss = self.reconstruction_loss(output, y_imgs)
            sem_const_loss = self.semantic_consistency_loss(output, y_imgs)
            # tex_const_loss = self.texture_consistency_loss(output, y_imgs)
            # tex_const_loss = 0
            adv_loss = -self.discriminator(output).mean()

            # g_loss = self.opt.OPTIM.RECON * recon_loss + \
            #          self.opt.OPTIM.SEMANTIC * sem_const_loss + \
            #          self.opt.OPTIM.TEXTURE * tex_const_loss + \
            #          self.opt.OPTIM.ADVERSARIAL * adv_loss

            g_loss = self.opt.OPTIM.RECON * recon_loss + \
                     self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                     self.opt.OPTIM.ADVERSARIAL * adv_loss
            g_loss.backward()
            self.optimizer_joint.step()
            self.wandb.log({"recon_loss": recon_loss.item(),
                            "sem_const_loss": sem_const_loss.item(),
                            # "tex_const_loss": tex_const_loss.item(),
                            "adv_loss": adv_loss.item()}, commit=False)

            info += "D Loss: {} ".format(d_loss)
            info += "G Loss: {} ".format(g_loss)
            print("End of embracing the model")
            if self.num_step % self.opt.MODEL.RAINDROP_LOG_INTERVAL == 0:
                log.info(info)
                print("Logging and visualization of RAINDROP_LOG_INTERVAL a ")
            if self.num_step % self.opt.MODEL.RAINDROP_VISUALIZE_INTERVAL == 0:
                print("Logging and visualization of RAINDROP_LOG_INTERVAL b ")
                idx = self.opt.WANDB.NUM_ROW

                # Print shape of images
                print("Shape of original image:", y_imgs[idx].shape)
                print("Shape of masked image:", imgs[idx].shape)
                print("Shape of predicted masks:", pred_masks[idx].shape)
                print("Shape of output:", output[idx].shape)

                # Print content of images (for the first pixel)
                print("Content of original image (first pixel):", y_imgs[idx][0, 0, 0])
                print("Content of masked image (first pixel):", imgs[idx][0, 0, 0])
                print("Content of predicted masks (first pixel):", pred_masks[idx][0, 0])
                print("Content of output (first pixel):", output[idx][0, 0, 0])

                # Plot the images
                fig, axes = plt.subplots(1, 4, figsize=(15, 5))
                axes[0].imshow(self.to_pil(y_imgs[idx].cpu()))
                axes[0].set_title("Original Image")
                axes[1].imshow(self.to_pil(linear_unscaling(imgs[idx]).cpu()))
                axes[1].set_title("Masked Image")
                axes[2].imshow(self.to_pil(pred_masks[idx].cpu()))
                axes[2].set_title("Predicted Masks")
                axes[3].imshow(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()))
                axes[3].set_title("Output")
                plt.show()

                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(imgs[idx]).cpu()), caption="masked_image"),
                    self.wandb.Image(self.to_pil(pred_masks[idx].cpu()), caption="predicted_masks"),
                    self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                ]}, commit=False)

                # code to  Save images locally
                print("Code to save images locally started...")
                save_dir = "F:/IIT_final_yr/Myproject/FypModel/lastTest/project8copies/proj8test2/training_images"
                os.makedirs(save_dir, exist_ok=True)
                original_img_path = os.path.join(save_dir, f"original_{self.num_step}.png")
                masked_img_path = os.path.join(save_dir, f"masked_{self.num_step}.png")
                predicted_masks_path = os.path.join(save_dir, f"predicted_masks_{self.num_step}.png")
                output_path = os.path.join(save_dir, f"output_{self.num_step}.png")
                self.to_pil(y_imgs[idx].cpu()).save(original_img_path)
                self.to_pil(linear_unscaling(imgs[idx]).cpu()).save(masked_img_path)
                self.to_pil(pred_masks[idx].cpu()).save(predicted_masks_path)
                self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()).save(output_path)
                print("Code to save images locally ended...")
                # code to  Save images locally

            self.wandb.log({})
            if self.num_step % self.opt.MODEL.RAINDROP_SAVE_INTERVAL == 0 and self.num_step != 0:
                print("Logging and visualization of RAINDROP_LOG_INTERVAL c ")
                self.do_checkpoint(self.num_step)
                print("Logging and visualization of RAINDROP_LOG_INTERVAL d ")
        print("RaindropTrainer run method ended")
    def do_checkpoint(self, num_step):
        print("do_checkpoint method started")
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

        checkpoint = {
            'num_step': num_step,
            'mpn': self.mpn.state_dict(),
            'rin': self.rin.state_dict(),
            'D': self.discriminator.state_dict(),
            'optimizer_joint': self.optimizer_joint.state_dict(),
            'optimizer_D': self.optimizer_discriminator.state_dict(),
            # 'scheduler_mpn': self.scheduler_mpn.state_dict(),
            # 'scheduler_rin': self.scheduler_rin.state_dict(),
            # 'scheduler_joint': self.scheduler_joint.state_dict(),
            # 'scheduler_D': self.scheduler_discriminator.state_dict(),
        }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))
        print("do_checkpoint method ended")

    def load_checkpoints(self, num_step):
        print("load_checkpoints method started")
        try:
            # print("load checkpint weight"+ self.opt.MODEL.RAINDROP_WEIGHTS)
            checkpoints = torch.load(self.opt.MODEL.RAINDROP_WEIGHTS)
            print("load checkpint 1")
            self.num_step = checkpoints["num_step"]
            print("load checkpint 2")
            self.mpn.load_state_dict(checkpoints["mpn"])
            print("load checkpint 3")
            self.rin.load_state_dict(checkpoints["rin"])
            print("load checkpint 4")
            self.discriminator.load_state_dict(checkpoints["D"])
            print("load checkpint 5")
            self.optimizers_to_cuda()
        except Exception as e:
            # Nontype error occurs here when the checkpoint is not loaded properly
            print("Error loading checkpoints:", e)
        print("load_checkpoints method ended")
    # chage this
    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.mpn = torch.nn.DataParallel(self.mpn).cuda()
            self.rin = torch.nn.DataParallel(self.rin).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.mpn = self.mpn.cuda()
            self.rin = self.rin.cuda()
            self.discriminator = self.discriminator.cuda()

# chage this
    def optimizers_to_cuda(self):
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_joint.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
