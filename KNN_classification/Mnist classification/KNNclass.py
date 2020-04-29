import numpy as np

class knn:
    def __init__(self, k, x, y):
        self.k = k
        self.x = x
        self.y = y

    def show(self):
        print("k :", self.k)
        print("x :", self.x)
        print("y :", self.y)

    # Calculate the distance btw test data and train data sets
    def cal_distance(self, input_x):
        distances = []

        i = 0
        while i < len(self.x):
            distance = np.linalg.norm(input_x - self.x[i])
            distance = distance ** (1/2)
            dis_pair = [[distance, i]]
            distances += dis_pair
            i += 1
        return distances

    # Obtain k classes : 정렬하여 상위 k개를 list slicing을 이용하여 추출
    def obtain_KNN(self, distances):
        distances.sort()
        idx_list = []
        dis_list = []

        j = 0
        for i in distances[:self.k]:
            idx_list.append(self.y[i[1]])
            dis_list.append(distances[j][0])
            j += 1
        return idx_list, dis_list

    # Obtain class with Majority Vote method : class_list의 값 중 어떠한 값이 가장 많은지를 조건문을 통해 파악하여 반환하고 카운팅은 +1로 한다. class_num = class의 개수
    def obtain_mv(self, class_list, class_num):
        count = np.zeros(class_num)

        for i in class_list:
            count[i] += 1

        return np.argmax(count)

    # Obtain class with Weighted Majority Vote method : class_list의 값 중 어떠한 값이 가장 많은지를 조건문을 통해 파악하여 반환하고 카운팅은 1/distance로 한다. class_num = class의 개수
    def obtain_wmv(self, input_x, class_num):
        distances = self.cal_distance(input_x)
        class_list, dis_list = self.obtain_KNN(distances)
        count = np.zeros(class_num)

        j = 0
        for i in class_list:
            count[i] += 1 / (dis_list[j] + np.finfo(float).eps)
            j += 1

        return np.argmax(count)

    def get_score(self, output, answer):
        if output == answer:
            return 1
        else:
            return 0
