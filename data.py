import numpy as np
import torch
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from criterion import calculate_i_index, calculate_m_index

dataset_path = '/Users/a18651298/Desktop/data/'
STRICT_PATH = 'strict_filter.pth'

back_of_head_channel = 0


def load_dataset():
    data = np.zeros(shape=(1099, 40, 1024))
    file_names = []
    exclude_list = ['src', '.DS_Store', '.ipynb_checkpoints']
    dir_list = os.listdir(dataset_path)
    dir_list = [i for i in dir_list if i not in exclude_list]
    count = 0
    for file_name in dir_list:
        data[count] = torch.load(os.path.join(dataset_path, file_name)).numpy()
        file_names.append(os.path.join(dataset_path, file_name))
        count += 1
    return data, file_names


def iqr_filter(values_list, strict=False):
    if strict:
        koef = 1.5
    else:
        koef = 3
    result = list(values_list)
    q_1 = np.percentile(values_list, 25)
    q_3 = np.percentile(values_list, 75)
    iqr = q_3 - q_1
    x_down = q_1 - koef * iqr
    x_up = q_3 + koef * iqr
    for value in result:
        if value > x_up or value < x_down:
            result.remove(value)
    return np.array(result)


def filter_and_mean(tensor, strict_filtering=False):
    # tensor = abs(np.fft.fft(tensor, axis=1))
    mean_values = np.zeros(shape=tensor.shape[0])
    for channel_num in range(tensor.shape[0]):
        values = iqr_filter(tensor[channel_num], strict=strict_filtering)
        mean_values[channel_num] = np.mean(values)
    return mean_values


def find_outliers_and_means(word_dataset, strict_filter=False):
    num_word = word_dataset.shape[0]
    num_channels = word_dataset.shape[1]
    processed_data = np.zeros(shape=(num_word, num_channels))
    print('Getting rid of outliers:')
    for tensor_number in range(num_word):
        print(str(tensor_number + 1) + ' / ' + str(num_word))
        processed_data[tensor_number] = filter_and_mean(word_dataset[tensor_number], strict_filtering=strict_filter)
    return processed_data


def form_cluster_members(data, distr, number_of_clusters, subclustering=False):
    result = []
    for cluster_number in range(number_of_clusters):
        cluster_power = list(distr).count(cluster_number)  # number of elements in cluster
        if subclustering:
            res = np.zeros(shape=cluster_power)
        else:
            res = np.zeros(shape=(cluster_power, 40))
        count = 0
        for index in range(len(distr)):
            if distr[index] == cluster_number:
                res[count] = data[index]
                count += 1
        result.append(res)
    return result  # list of numpy arrays of different sizes


def launch_k_means(data, k_clusters, subclusters=False):
    k_means = KMeans(n_clusters=k_clusters)
    k_means.fit(data)
    cluster_members = form_cluster_members(data, k_means.fit_predict(data), k_clusters, subclustering=subclusters)
    return cluster_members

# На вход алгоритм получает список кластеров с их элементами - векторами длины 40, то есть результат работы алгоритма
# кластеризации


def classify_clusters(cluster_members):
    cluster_class = np.zeros(shape=(len(cluster_members)))  # создание массивов с результатами классификации
    elements_classes = []
    for cluster_num in range(len(cluster_members)):  # итерация по всем кластерам разбиения
        element_class = np.zeros(shape=len(cluster_members[cluster_num]))
        special_channel_data = np.reshape(np.array([cluster_members[cluster_num][j][back_of_head_channel]
                                                    for j in range(cluster_members[cluster_num].shape[0])]),
                                          newshape=(-1, 1))
        # мы спроектировали вектора длины 40 на одну их координату - с номером back_of_head_channel
        subclassed_members = launch_k_means(special_channel_data, 2, subclusters=True)
        # разбиваем кластер на два подкластера методом k-средних; subclassed_members[i] - полученные подкластеры
        mean_cluster_0 = np.mean(subclassed_members[0])  # находим среднее значение элементов каждого подкластера
        mean_cluster_1 = np.mean(subclassed_members[1])
        if mean_cluster_0 >= mean_cluster_1:  # если первый подкластер выше второго
            if len(subclassed_members[0]) > len(subclassed_members[1]):  # если в первом подкластере больше элементов
                cluster_class[cluster_num] = 0  # то весь кластер зашумлённый
            else:
                cluster_class[cluster_num] = 1  # иначе весь кластер основной
            for element in range(len(cluster_members[cluster_num])):  # итерация по всем элементам кластера
                if element < len(subclassed_members[0]):  # элементы из зашумлённого подкластера - низкого качества
                    element_class[element] = 0
                else:     # элементы из основного подкластера - высокого качества
                    element_class[element] = 1
        else:  # если второй подкластер выше первого
            if len(subclassed_members[0]) > len(subclassed_members[1]):  # если в первом подкластере больше элементов
                cluster_class[cluster_num] = 1  # то весь кластер основной
            else:
                cluster_class[cluster_num] = 0  # иначе весь кластер зашумлённый
            for element in range(len(cluster_members[cluster_num])):  # итерация по всем элементам кластера
                if element < len(subclassed_members[0]):  # элементы из основного подкластера - высокого качества
                    element_class[element] = 1
                else:   # элементы из зашумлённого подкластера - низкого качества
                    element_class[element] = 0
        elements_classes.append(element_class)
    return cluster_class, elements_classes


if __name__ == "__main__":
    # dataset, files = load_dataset()
    # processed_dataset = find_outliers_and_means(dataset, strict_filter=True)
    # torch.save(torch.from_numpy(processed_dataset), STRICT_PATH)
    dataset = torch.load(STRICT_PATH).numpy()
    print(dataset.shape)
    k_arr = np.zeros(shape=(31-2))
    N_arr = np.zeros(shape=(31-2))
    for k in range(2, 31):
        mean = 0
        print('k = ', k)
        for exp_num in range(20):
            cluster_distribution = launch_k_means(dataset, k)
            for i in range(k):
                print(len(cluster_distribution[i]))
            cluster_classes, elements_class = classify_clusters(cluster_distribution)
            print(cluster_classes)
            N_index = calculate_i_index(cluster_distribution, cluster_classes, elements_class)
            mean += N_index
        mean /= 20
        print(mean)
        k_arr[k-2] = k
        N_arr[k-2] = mean
    plt.plot(k_arr, N_arr)
    plt.title("График зависимости индекса I от количества кластеров разбиения")
    plt.xlabel("Количество кластеров k")
    plt.ylabel("I - индекс")
    plt.show()
