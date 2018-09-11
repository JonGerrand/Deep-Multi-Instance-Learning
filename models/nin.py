from network import Network

class NIN(Network):
    def setup(self):
        if self.num_classes is not None:
            num_classes = self.num_classes
        else:
            num_classes = 1000
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .conv(1, 1, 96, 1, 1, name='cccp1')
             .conv(1, 1, 96, 1, 1, name='cccp2')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(5, 5, 256, 1, 1, name='conv2')
             .conv(1, 1, 256, 1, 1, name='cccp3')
             .conv(1, 1, 256, 1, 1, name='cccp4')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(1, 1, 384, 1, 1, name='cccp5')
             .conv(1, 1, 384, 1, 1, name='cccp6')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 1024, 1, 1, name='conv4-1024')
             .conv(1, 1, 1024, 1, 1, name='cccp7-1024')
             .conv(1, 1, 1000, 1, 1, name='cccp8-1024')
             .avg_pool(6, 6, 1, 1, padding='VALID', name='pool4')
             .fc(num_classes, relu=False, name='logits')
             .softmax(name='loss'))
