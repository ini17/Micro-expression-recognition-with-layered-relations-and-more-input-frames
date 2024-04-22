import pandas as pd


a = {
    'negative': 0,
    'others': 1,
    'positive': 2,
    'surprise': 3
}

b = [69, 94, 35, 56]
c = zip(a, b)
print("Total count:" + "")
print("%s: %d" % (list(a.keys())[i], b[i]) for i in range(4))

df = pd.read_excel("test_result.xlsx", header=0, sheet_name=0, index_col=0)
row_num = df.shape[0]
for x, y in c:
    df.loc[row_num, x] = y
    print(x, y)
df.to_excel("test_result.xlsx")
