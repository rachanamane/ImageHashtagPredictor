
from shared.features import ImageHashtagFeatures

class SingleImageObject:

    def __init__(self, features_dict):
        self.width = features_dict[ImageHashtagFeatures.widthFeature]
        self.height = features_dict[ImageHashtagFeatures.heightFeature]
        self.image_raw = features_dict[ImageHashtagFeatures.imageRawFeature]
        self.labels = features_dict[ImageHashtagFeatures.labelsFeature]