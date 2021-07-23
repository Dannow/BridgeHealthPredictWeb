from ShortTermForecast.DataPreProcess import *
import seaborn as sns

# 合并所有表
def MergeTable(ExcelName):
    # 获得总表
    df = pd.read_excel(ExcelName, sheet_name=None, index_col="Unnamed: 0")
    # 获得总表中的子表
    key = list(df.keys())
    # 创建空的dataFrame
    allData = pd.DataFrame()
    # 保存列名
    columnNames = []
    for i in key:
        # 获取字表原始数据
        allOriginalData = MissingValueProcessing(ExcelName, i)
        # allData = pd.merge(allOriginalData,allData,left_index=True,right_index=True,how="outer")
        # 获得每个字表新的列名
        columnNames += [i+b for b in allOriginalData.columns]
        # 拼接全部字表数据
        allData = pd.concat([allData, allOriginalData], axis=1, ignore_index=True)
    # 跟换列名
    allData.columns = columnNames
    return allData

# 绘制皮尔森热图
def drawPers(data, dataName):
    colormap = plt.cm.viridis
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of '+ dataName, y=1.05, size=15)
    sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)
    plt.savefig('./' + dataName + '.png')
    plt.show()


# 绘制多表皮尔森热图
def drawPersPicMore(ExcelName):
    allData = MergeTable(ExcelName)
    drawPers(allData, 'All')


# 绘制单个皮尔森热图
def drawPersPic(ExcelName, sheetName):
    allOriginalData = MissingValueProcessing(ExcelName, sheetName)
    drawPers(allOriginalData, sheetName)

