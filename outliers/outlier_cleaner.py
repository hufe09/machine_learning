#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    # your code goes here
    for i in range(len(predictions)):
        cleaned_data.append((ages[i], net_worths[i],
                             abs(predictions[i] - net_worths[i])))
    # 根据错误值排序
    cleaned_data.sort(key=lambda x: x[2])

    # 删除具有最大残差的 10% 的点
    return cleaned_data[:int(len(ages) * 0.9)]


if __name__ == "__main__":
    predictions = [1, 2, 3, 4, 5]
    ages = [1, 2, 3, 4, 5]
    net_worths = [3, 4, 6, 7, 3, 2]
    cleaned_data = outlierCleaner(predictions, ages, net_worths)
    print(cleaned_data)
