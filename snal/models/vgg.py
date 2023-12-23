'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def extract_layer(self, x):
        lays = []
        features = []
        for layer in self.layers:
            lays += layer
            mod = nn.Sequential(*lays)
            f = mod(x)
            features.append(f)
        
        return features
            

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            
                           
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        
        self.layers = layers
        self.layers_n = len(layers)
        
        return nn.Sequential(*layers)
    
    def predict_prob(self, x):
            y = self.forward(x)
            prob = F.softmax(y, dim = 1)
            return prob

    def distribute_forward(self, x1, x2):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return y1, y2





def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()


class VGG_cifar100(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_cifar100, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        num_class = 100
        self.num_class = num_class
        self.dropout = nn.Dropout(p = 0.25)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def predict_prob(self, x):
        y = self.forward(x)
        prob = F.softmax(y, dim = 1)
        return prob

    def distribute_forward(self, x1, x2):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return y1, y2
        
    def forward_dropout(self, x):
        out = self.features(x)
        #print(out.size)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.dropout(out)
        return out
        
    def predict_prob_dropout(self, x):
        y = self.forward_dropout(x)
        prob = F.softmax(y, dim = 1)
        return prob
        


class VGG_cifar10(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_cifar10, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        num_class = 10
        self.num_class = num_class
        self.dropout = nn.Dropout(p = 0.25)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #print(out.size)
        out = self.classifier(out)
        return out


    def extract_feature(self, x):
        lays = []
        features = []
        for layer in self.layers:
            lays += [layer]
            mod = nn.Sequential(*lays)
            f = mod(x)
            features.append(f)
        
        return features

    
         
        
        
    
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        
        self.layers = layers
        print(self.layers)
        
        return nn.Sequential(*layers)
    
    def forward_dropout(self, x):
        out = self.features(x)
        #print(out.size)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.dropout(out)
        return out
        
    def predict_prob_dropout(self, x):
        y = self.forward_dropout(x)
        prob = F.softmax(y, dim = 1)
        return prob
    
    def predict_prob(self, x):
        y = self.forward(x)
        prob = F.softmax(y, dim = 1)
        return prob
        
    
    
    def distribute_forward(self, x1, x2):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return y1, y2


class VGG_mnist(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_mnist, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        num_class = 10
        self.classifier = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def predict_prob(self, x):
        y = self.forward(x)
        prob = F.softmax(y, dim = 1)
        return prob

    def distribute_forward(self, x1, x2):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return y1, y2

 
 
class VGG_caltech101(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_caltech101, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.features_ = self._make_layers_(cfg[vgg_name])
        num_class = 102
        self.num_class = num_class
        self.classifier = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_class)
        )
        self.dropout = nn.Dropout(p = 0.25)
        
        self.classifier_ = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_class)
        )
       
       
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    
    
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)] ##1 sampling &2 training
        return nn.Sequential(*layers)
    
    def _make_layers_(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=3)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
    
    def predict_prob(self, x):
        y = self.forward(x)
        prob = F.softmax(y, dim = 1)
        return prob

    def distribute_forward(self, x1, x2):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return y1, y2
 
    def forward_dropout(self, x):
        out = self.features_(x)
        #print(out.size)
        out = out.view(out.size(0), -1)
        out = self.classifier_(out)
        out = self.dropout(out)
        return out
        
    def predict_prob_dropout(self, x):
        y = self.forward_dropout(x)
        prob = F.softmax(y, dim = 1)
        return prob
    


class VGG_caltech256(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_caltech256, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        num_class = 258
        self.num_class = num_class
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, num_class)
        )


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def forward_dropout(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.dropout(out)
        return out
        
        

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=3)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def predict_prob(self, x):
        y = self.forward(x)
        prob = F.softmax(y, dim = 1)
        return prob

    def predict_prob_dropout(self, x):
        y = self.forward_dropout(x)
        prob = F.softmax(y, dim = 1)
        return prob
        
        
        

    def distribute_forward(self, x1, x2):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return y1, y2

