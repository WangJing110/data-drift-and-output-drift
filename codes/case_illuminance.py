import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


class BRIGHT():
    def brightness_v1(self,file):
        image = cv2.imread(filename=file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.mean(image)[0]

    def brightness_v2(self,file):
        image = cv2.imread(filename=file)
        mean = cv2.mean(image)
        R, G, B = mean[0], mean[1], mean[2]
        return math.sqrt(0.241*(R**2) + 0.691*(G**2) + 0.068*(B**2))

    def fetch_filenames(self,path):
        filenames = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file[-4:] == ".png":
                    filename = os.path.join(root, file)
                    filenames.append(filename)
        return filenames

    def calculate_brightness(self,path1,path2):
        filenames1 = self.fetch_filenames(path1)
        filenames2 = self.fetch_filenames(path2)

        brightness1 = []
        brightness2 = []

        for filename in filenames1:
            brightness1.append(self.brightness_v1(filename))
        for filename in filenames2:
            brightness2.append(self.brightness_v1(filename))

        plt.hist(x=brightness1, bins=50, density=True, stacked=True,color="blue", alpha=0.7)
        plt.hist(x=brightness2, bins=50, density=True, stacked=True,color="yellow", alpha=0.7)
        plt.xlabel("Brightness")
        # plt.suptitle("Input PSI: " + str(0.0507), fontsize=14, fontweight='bold')
        plt.show()
        # print("Train China, (mean, std):" + str(norm.fit(brightness1)))
        # print("Test China, (mean, std):" + str(norm.fit(brightness2)))

        return brightness1,brightness2


class PSI():
    def scale_range(self,input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    def sub_psi(self,e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return (value)

    def calculate_psi(self,expected_array, actual_array, buckets=10, isrange=False):
        psi_values = np.empty(expected_array.shape)
        if isrange:
            for i in range(0, len(expected_array)):
                psi_values[i] = self.sub_psi(expected_array[i], actual_array[i])
        else:
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            minv, maxv = 0,256
            breakpoints = self.scale_range(breakpoints, minv, maxv)

            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

            psi_values = np.empty(actual_percents.shape)
            for i in range(0, len(expected_percents)):
                psi_values[i] = self.sub_psi(expected_percents[i], actual_percents[i])

        # print("psi_values：",psi_values)
        psi_value = np.sum(psi_values)

        return (psi_value)


if __name__ == "__main__":
    bri = BRIGHT()
    psi = PSI()

    #input validation
    path_train_data = "../dataset/showcase/china_original/test"
    path_test_data = "../dataset/showcase/china_dark/test"
    brightness_list_train, brightness_list_test = bri.calculate_brightness(path_train_data,path_test_data)
    #
    brightness_train, brightness_test = np.array(brightness_list_train), np.array(brightness_list_test)
    input_psi = psi.calculate_psi(brightness_train,brightness_test,10)
    print("brightness train_" , "MAX: " , max(brightness_train) ,
          "MIN: " , min(brightness_train) , "MEAN: " , np.mean(brightness_train) ,
          "COUNT: " , str(len(brightness_list_train)))
    print("brightness test_" , "MAX: " , max(brightness_test) ,
          "MIN: " , min(brightness_test) , "MEAN: " , np.mean(brightness_test) ,
          "COUNT: " , str(brightness_test.shape))
    print("input_drift_psi:", input_psi)

    #output validation

    # different brightness
    d1 = np.array([0.1728, 0.22706667, 0.20666667, 0.19213333, 0.20133333])
    d2 = np.array([0.32      , 0.12266667, 0.23506667, 0.10946667, 0.2128    ])
    # d2 = np.array([0.16173333, 0.2356, 0.2116, 0.1884, 0.20266667])

    # print("model accuracy:", 0.97)
    print("normal brightness dataset output:",d1)
    print("dark dataset output:",d2)
    #
    ouput_psi = psi.calculate_psi(d1, d2, 10, True)
    #
    print("output_psi:",ouput_psi)

    plt.bar(range(len(d1)), list(d1), width=0.6, alpha=0.7)
    plt.bar(range(len(d2)), list(d2), width=0.6, alpha=0.7)
    plt.suptitle("China & China Very Dark Classification Result Distribution", fontsize=14, fontweight='bold')
    plt.show()

