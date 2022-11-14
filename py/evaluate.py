import re
import argparse

import numpy as np
from obsHandle import OBSHandler


def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="evaluation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pred_path', type=str, help='y_pred obs path')
    parser.add_argument('--true_path', type=str, help='y_true obs path')
    parser.add_argument('--cls', default=256, type=int, help='the nums of cls')
    parser.add_argument('--pos', default=1, type=int, help='the first pos of cls')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt


def evaluate4cls(pred_path, y_true_path, cls, pos=1):
    obs = OBSHandler()
    data_pred = obs.readFile(pred_path)["content"]
    data_true = obs.readFile(y_true_path)["content"]
    obs.close_obs()
    digit_regex = r"\d+"
    pattern = re.compile(digit_regex)
    y_pred = list(map(lambda x: int(x), pattern.findall(data_pred)))
    y_true = list(map(lambda x: int(x), pattern.findall(data_true)))
    if len(y_pred) != len(y_true):
        return {
            "status": -1,
#             "metrics": None,
            "msg": "评估失败，结果长度不一致"
        }
    ec = Evaluation4Classfication(y_pred, y_true, cls=cls, pos=pos)
    return {
        "status": 200,
        "metrics": ec.evaluate(),
        "msg": "评估成功"
    }


class Evaluation4Classfication:
    def __init__(self, y_pred, y_true, cls, pos=0):
        """
        :param y_pred: 为预测label值列表
        :param y_true: 为真实label值列表
        :param cls: 为总类别数
        :param pos: 起始位置，default：0
        """
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)
        self.cls = cls
        self.pos = pos
        self.total = len(y_true)
        self.acc = np.sum(self.y_pred == self.y_true) / self.total if self.total != 0 else 0
        self.err = 1 - self.acc

    def getPRF4Onecls(self, cls, beta=None):
        """
        :param cls: 当前类别
        :param beta:
            beta=0，precsion和recall同等重要；
                 0.5，recall的重要程度是precsion的一半；
                 2，recall的重要程度是precsion的一半
        :return:
        """
        # 计算TP、FP、FN、TN
        y_pred_cls = self.y_pred == cls
        y_true_cls = self.y_true == cls
        t_total = np.sum(y_true_cls)
        f_total = self.total - t_total
        p_total = np.sum(y_pred_cls)
        n_total = self.total - p_total

        tp = np.sum(y_pred_cls[y_true_cls == True])
        fp = p_total - tp
        fn = t_total - tp
        tn = f_total - fp

        # 计算precision、recall、f1、f_beta
        precison = tp / p_total if p_total != 0 else 0
        recall = tp / t_total if t_total != 0 else 0
        if beta == None:
            beta = [1]
        f1 = 2 * precison * recall / (precison + recall) if (precison + recall) != 0 else 0
        beta = 0.5
        f05 = (1 + beta * beta) * precison * recall / (beta * beta * precison + recall) \
            if (beta * beta * precison + recall) != 0 else 0
        beta = 2
        f2 = (1 + beta * beta) * precison * recall / (beta * beta * precison + recall) \
            if (beta * beta * precison + recall) != 0 else 0

        return precison, recall, f1, f05, f2

    def getAveragePRF(self, beta=None):
        ap, ar, af1, af05, af2 = 0, 0, 0, 0, 0
        for i in range(self.pos, self.cls + self.pos):
            precison, recall, f1, f05, f2 = self.getPRF4Onecls(cls=i, beta=beta)
            ap += precison
            ar += recall
            af1 += f1
            af05 += f05
            af2 += f2
        return ap / self.cls, ar / self.cls, af1 / self.cls, af05 / self.cls, af2 / self.cls

    def evaluate(self):
        ap, ar, af1, af05, af2 = self.getAveragePRF()
        return {
            "ap": ap,
            "ar": ar,
            "af1": af1,
            "af05": af05,
            "af2": af2,
            "acc": self.acc,
            "err": self.err
        }


if __name__ == "__main__":
    """
    python evaluate.py --y_pred_path xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/submit_result/s9qfqri3zpc8j2x7_1/result_example_5120-2022-8-8-15-3-16.txt 
                       --y_true_path xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/result/label.txt
                       --cls 256
                       --pos 1
    """
    args_opt = parse_args()
    y_pred_path = args_opt.pred_path
    y_true_path = args_opt.true_path
    cls = args_opt.cls
    pos = args_opt.pos
    # y_pred_path = "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/submit_result/s9qfqri3zpc8j2x7_1/result_example_5120-2022-8-8-15-3-16.txt"
    #
    # cls = 256
    # pos = 1
    res = evaluate4cls(y_pred_path, y_true_path, cls, pos=1)
    print(res)
