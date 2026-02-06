class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val   # 变量
        self.sum += val * n  # 变量累计和
        self.count += n  # 变量累计次数
        self.avg = self.sum / self.count  # 变量累计均值
