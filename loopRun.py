from train import main


num = 0
acc, f1 = 0, 0
max_acc, max_f1 = 0, 0
for i in range(20):
    num += 1
    ret = main()
    acc += ret[0]
    f1 += ret[1]
    if ret[0] > max_acc:
        max_acc = ret[0]
        max_f1 = ret[1]
print(f"Average Accuracy: {acc / num: .4f}, F1-Score: {f1 / num: .4f}")
print(f"Max Accuracy: {max_acc: .4f}, Max F1-Score: {max_f1: .4f}")
# while True:
#     ret = main()
#     if ret[0] > 96.33:
#         print(f"Accuracy: {ret[0]: .4f}, F1-Score: {ret[1]: .4f}")
#         break
