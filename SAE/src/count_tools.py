from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score

def macro_acc(true_label, pre_label):
    label_2_num=defaultdict(int)
    label_pre_true_num=defaultdict(int)
    label_2_acc=defaultdict(float)

    sz=len(true_label)
    for i in range(sz):
        label_2_num[true_label[i]]+=1
        if(pre_label[i]==true_label[i]):
            label_pre_true_num[true_label[i]]+=1
    for label,num in label_2_num.items():
        label_2_acc[label]=float(label_pre_true_num[label]/num)
    sum=0.

    for i,j in label_2_acc.items():
        sum+=j

    return sum/len(label_2_acc)

if __name__ == "__main__":
    real_label_test, pre_label_test_1=[1,1,2,2,3,3,4,4,5,6,7,3,2,9,7,5,4,2],[2,2,3,3,1,1,1,4,3,5,2,6,7,7,2,2,3,3]
    ans=0.25
    pre=macro_acc(real_label_test,pre_label_test_1)
    pre2=accuracy_score(real_label_test,pre_label_test_1)
    if(pre==ans):
        print("success!")
        print(pre)
    else:
        print("fail")
        print(pre)
        print(pre2)


