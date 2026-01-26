import cv2 as cv
import torch
from torchvision.transforms import v2

from smhc import calculate_orientation_maps


class DifferenceOfGaussiansImageConfiguration:
    def __init__(self, args) -> None:
        self.kernel_size = args.kernel_size
        self.highpass_sigma = args.highpass_sigma
        self.lowpass_sigma = args.lowpass_sigma


class GaborFilterBankConfiguration:
    def __init__(self, args) -> None:
        self.kernels_num = args.gb_kernels_num
        self.kernel_size = args.gb_kernel_size

        self.lamb_da = args.gb_lambda
        self.phase = args.gb_phase
        self.sigma_x = args.gb_sigma_x
        self.sigma_y = args.gb_sigma_y


class GaborFilterBank:
    def __init__(self, args) -> None:
        self.dog_image_config = DifferenceOfGaussiansImageConfiguration(args)
        self.filter_bank_config = GaborFilterBankConfiguration(args)
        self.acquire_all_gabor_kernels()

    def acquire_all_gabor_kernels(self):
        """
        Acquire the complete gabor filter bank and store it in a 1-D tensor
        """
        kernel_size_2 = (
            self.filter_bank_config.kernel_size * self.filter_bank_config.kernel_size
        )

        all_gb_kernels_elements = self.filter_bank_config.kernels_num * kernel_size_2

        all_gb_kernels = torch.zeros(all_gb_kernels_elements, dtype=torch.float32)

        for k in range(self.filter_bank_config.kernels_num):
            theta = k * torch.pi / self.filter_bank_config.kernels_num

            gb_kernel = cv.getGaborKernel(
                (
                    self.filter_bank_config.kernel_size,
                    self.filter_bank_config.kernel_size,
                ),
                self.filter_bank_config.sigma_x,
                theta,
                self.filter_bank_config.lamb_da,
                self.filter_bank_config.sigma_x / self.filter_bank_config.sigma_y,
                self.filter_bank_config.phase,
                ktype=cv.CV_32F,
            )
            gb_kernel_flatten = gb_kernel.flatten()
            all_gb_kernels[k * kernel_size_2 : (k + 1) * kernel_size_2] = (
                torch.from_numpy(gb_kernel_flatten)
            )

        self.all_gb_kernels = all_gb_kernels.cuda()

    def acquire_orientation_maps(self, image):
        """
        Calculate the 2d orientation and confidence maps
        """
        # bandpass_filter
        high_pass_filter = v2.GaussianBlur(
            kernel_size=self.dog_image_config.kernel_size,
            sigma=self.dog_image_config.highpass_sigma,
        )
        img_hp = high_pass_filter(image.unsqueeze(0))
        low_pass_filter = v2.GaussianBlur(
            kernel_size=self.dog_image_config.kernel_size,
            sigma=self.dog_image_config.lowpass_sigma,
        )
        img_lp = low_pass_filter(image.unsqueeze(0))
        img_bp = img_hp - img_lp

        ori2d, conf = calculate_orientation_maps(
            img_bp.squeeze(0).cuda(),
            self.all_gb_kernels,
            self.filter_bank_config.kernels_num,
            self.filter_bank_config.kernel_size,
        )

        return (ori2d, conf)
