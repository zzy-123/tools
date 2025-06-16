import numpy as np

"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
confusionMetric
P\\L    P    N

P      TP    FP

N      FN    TN

"""


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        # ACC = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def ReCall(self):
        # ReCall = TP / ( TP + FN )
        recall = np.diag(self.confusion_matrix) / (np.diag(self.confusion_matrix) +
                                                   np.sum(self.confusion_matrix, axis=1) - np.diag(
                    self.confusion_matrix))
        # recall = self.confusion_matrix[0, 0] / (self.confusion_matrix[0, 0] + self.confusion_matrix[1, 0])
        return recall

    def Precision(self):
        # Precision = TP / ( TP + FP )
        precision = np.diag(self.confusion_matrix) / (np.diag(self.confusion_matrix) +
                                                      np.sum(self.confusion_matrix, axis=0) - np.diag(
                    self.confusion_matrix))
        # precision = self.confusion_matrix[0, 0] / (self.confusion_matrix[0, 0] + self.confusion_matrix[0, 1])
        return precision

    def F1_score(self):
        # F1 = 2*recall*precision / (recall+precision)
        F1 = 2 * self.ReCall() * self.Precision() / (self.ReCall() + self.Precision())
        return F1

    def IoU(self):
        # IoU = TP / ( TP + FP + FN )
        iou = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        return iou

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 1, 0, 0]) # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 1, 0]) # 可直接换成标注图片
    metric = Evaluator(2) # 3表示有3个分类，有几个分类就填几
    metric.add_batch(imgLabel, imgPredict)
    pa = metric.Pixel_Accuracy()
    cpa = metric.classPixelAccuracy()
    Precision = metric.Precision()
    mIoU = metric.Mean_Intersection_over_Union()
    IoU = metric.IoU()
    recall = metric.ReCall()
    F1 = metric.F1_score()
    print('pa is : %f' % pa)
    print('cpa is :') # 列表
    print(cpa)
    print('mIoU is : %f' % mIoU)
    print('IoU is :') # 列表
    print(IoU)
    print('Precision is :') # 列表
    print(Precision)
    print('recall is :') # 列表
    print(recall)
    print('F1 is :') # 列表
    print(F1)
