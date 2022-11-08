import spixel.run_demo as spxiel
import saliency.basnet_test as saliency

class Spixel:
    def __init__(self, sp_model_dir, sp_save_dir):
        self.sp_model_dir = sp_model_dir
        self.sp_save_dir = sp_save_dir
        self.sp_model = spxiel.load_model(sp_model_dir)

    def do_spixel(self, img_path):
        label, sp_img = spxiel.test_single(self.sp_model, img_path)
        return label, sp_img

class Saliency:
    def __init__(self, sal_model_dir, sal_save_dir):
        self.sal_model_dir = sal_model_dir
        self.sal_save_dir = sal_save_dir
        self.sal_model = saliency.load_model(sal_model_dir)

    def saliency_detect(self, img_path):
        sal_imo = saliency.test_single(self.sal_model, img_path)
        return sal_imo
