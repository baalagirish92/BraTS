
import glob
import torch
import torchio as tio

class BrainTumorData():
    def __init__(self, image_folder,img_type,splits):
        super().__init__()
        self.image_folder = image_folder
        self.img_type = img_type
        self.splits=splits

    
    def get_image_lists(self):
        image_paths = sorted(glob.glob(self.image_folder+"\\*"+ "\\*"+self.img_type+".nii.gz"))
        label_paths = sorted(glob.glob(self.image_folder+"\\*"+ "\\*"+"seg.nii.gz"))
        assert len(image_paths) == len(label_paths)
        
        print(len(image_paths))
        return image_paths, label_paths
    
    
    def prepare_data(self):
        self.subjects = []
        image_paths, label_paths=self.get_image_lists()
        for (image_path, label_path) in zip(image_paths, label_paths):
            subject = tio.Subject(
                mri=tio.ScalarImage(image_path),
                brain=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)
        print('Dataset size:', len(self.subjects), 'subjects')

    def get_preprocessing_transform(self, type):
        
        if type=="train" or type=="valid": 
            preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad((128, 128, 128)),
            tio.RemapLabels({4:3}),
            tio.OneHot(num_classes=4),
            ])
    
        else:
            preprocess = tio.Compose([
             tio.ToCanonical(),
             tio.CropOrPad((128, 128, 128)),
             tio.RemapLabels({4:3}),
             ])
        return preprocess

    def setup(self):

        train_subjects, val_subjects, test_subjects = torch.utils.data.random_split(self.subjects, self.splits)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.get_preprocessing_transform("train"))
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.get_preprocessing_transform("valid"))
        self.test_set = tio.SubjectsDataset(test_subjects, transform=self.get_preprocessing_transform("train"))
