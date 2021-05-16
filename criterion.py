eps = 1e-3
N = 1099

# Мы считаем, что для каждого x_j из X известен вектор значений его признаков a = (a_0, a_2, ..., a_k-1), где k =|A| -
# число признаков, a_i может принимать значения от 0 до dom(a_i) - 1. dom = (dom(a_0), ..., dom(a_k-1)) - вектор из
# областей значений признаков.


def calculate_m_index(clusters, attr_values, dom):
    m = 0  # вводим переменную для итогового значения M-индекса
    for attribute in range(dom.shape[0]):  # итерация по всем имеющимся признакам
        sub_sum = 0  # вводим переменную для подсуммы
        for attr_value in range(dom[attribute]):  # итерация по всем значениям данного признака
            max_value = 0  # вводим переменную для максимума
            for cluster in range(len(clusters)):  # итерация по всем кластерам разбиения
                upper = 0  # вводим переменную для числителя дроби
                for element in range(clusters[cluster].shape[0]):  # итерация по всем элементам кластера
                    upper += int(attr_values[cluster][element][attribute] == attr_value)
                    # индекс увеличивается, если элемент имеет данное значение признака
                upper /= clusters[cluster].shape[0]  # нормализация числителя

                lower = 0  # вводим переменную для знаменателя дроби
                for another_cluster in range(len(clusters)):  # итерация по всем остальным кластерам, кроме данного
                    if cluster != another_cluster:
                        for element in range(clusters[another_cluster].shape[0]): # итерация по всем элементам кластера
                            lower += int(attr_values[another_cluster][element][attribute] == attr_value)
                            # индекс уменьшается, если элемент имеет такое же значение признака
                lower /= (N - clusters[cluster].shape[0])  # нормализация знаменателя
                lower += eps  # добавление eps гарантирует неравенство нулю знаменателя

                value = upper / lower  # вычисление дроби
                if value > max_value:
                    max_value = value
            sub_sum += max_value
        sub_sum /= dom[attribute]
        m += sub_sum
    return m


# Мы считаем, что все кластеры уже классифицированы на основные и зашумлённые. При этом
# cluster_class[cluster_num] = 0, если кластер зашумлён, и равен 1, если кластер основной.
# Каждый элемент выборки также классифицирован. element_class[cluster_num][element_num] = 0, если элемент
# низкого качества, и равен 1, если высокого качества.


def calculate_i_index(clusters, cluster_class, element_class, alpha=2 * eps, beta=eps):
    error = 0  # вводим переменную для итоговой ошибки
    for cluster in range(len(clusters)):  # итерация по всем кластерам разбиения
        sub_error = 0
        if cluster_class[cluster] == 1:  # если кластер основной
            for element in range(clusters[cluster].shape[0]):  # итерация по всем элементам кластера
                sub_error += alpha * int(element_class[cluster][element] == 0)
                # ошибка увеличивается, если элемент низкого качества
        else:  # если кластер зашумлённый
            for element in range(clusters[cluster].shape[0]):  # итерация по всем элементам кластера
                sub_error += beta * int(element_class[cluster][element] == 1)
                # ошибка увеличивается, если элемент высокого качества
        error += sub_error
    return error
