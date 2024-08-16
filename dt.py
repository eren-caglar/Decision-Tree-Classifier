# -*- coding: utf-8 -*-
"""
@author: Eren Çağlar
"""
from typing import List


class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    
    def __init__(self, max_depth: int, min_samples_split = 2 ): # minimum örnek sayısı 2 olarak baştan belirleyelim
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X: List[List[float]], y: List[int]):
        if (len(y) != 0) and (len(X) != 0):
            self.root = self._build_tree(X,y)
            
    
    def _best_split(self, X: List[List[float]], y: List[int], features): 
        gini = float('inf')
        best_feature = None
        threshold = None
        left_or_right = None
        # her bir özellik için medyan değerini hesapla devamında her birine ait gini değerini bul
        for feature in features:
            column = [row[feature] for row in X]
            length_of_column = len(column)
            column.sort()
            if length_of_column % 2:
                median1 = column[length_of_column // 2]
                median2 = column[length_of_column // 2 -1]
                median_of_column = (median1 + median2)/2
            else:
                median_of_column  = column[length_of_column // 2]
            #gini hesabı yap ve en uygun gini olan kolon ve degeri _split'e yolla
            #gini hesabını y'deki oranlara göre belirleyeceğimizden onu bölelim
            y_left = [y[i] for i in range(len(y)) if X[i][feature] <= median_of_column]
            y_right = [y[i] for i in range(len(y)) if X[i][feature] > median_of_column] 
            #bu değişkenler için gini değeri hesaplanır
            gini_for_y_left = self._calculate_gini(y_left)
            gini_for_y_right = self._calculate_gini(y_right)
            #kolonların gini değerine göre atama yapılır
            if(gini_for_y_left <= gini):
                gini = gini_for_y_left
                best_feature = feature
                threshold = median_of_column
                left_or_right = 'left'
                
            if(gini_for_y_right <= gini):
                gini = gini_for_y_right
                best_feature = feature
                threshold = median_of_column
                left_or_right = 'right'
        
        return best_feature, threshold, left_or_right #gini'de gerekebilir!
                
                
                
    def _split(self, X_for_split: List[List[float]], y_for_split: List[int], feature_for_split, threshold):
        # bölünecek matrisler oluşturlması
        left_X, right_X = [], []
        left_y, right_y = [], []
        # matrislerin bölünmesi
        for row_num,label in zip(X_for_split, y_for_split):
            if(row_num[feature_for_split]<=threshold):
                left_X.append(row_num)
                left_y.append(label)
            else:
                right_X.append(row_num)
                right_y.append(label)
        return left_X, right_X, left_y, right_y
                
    
    def _build_tree(self, X_for_tree: List[List[float]], y_for_tree: List[int], current_depth = 0):
        # Boş liste kontrolü
        if len(y_for_tree) == 0:
            return None
        # y'deki elemanların hepsi aynı türden mi? flag ile kontrol et
        first_value_of_y = y_for_tree[0]
        flag = 1
        for element in y_for_tree:
            if first_value_of_y != element:
                flag = 0
                break
        # toplam örnek sayısı
        num_samples = len(y_for_tree)
        # Durdurma kriterlerinin kontrolü
        if(current_depth >= self.max_depth or flag == 1 or num_samples < self.min_samples_split):
            predicted_class = self._most_common_label(y_for_tree)
            return Node(value = predicted_class)
        features = list(range(len(X_for_tree[0])))
        best_feature, threshold, left_or_right = self._best_split(X_for_tree, y_for_tree, features )
        # verilerin bölünmesi
        left_X, right_X, left_y, right_y = self._split(X_for_tree, y_for_tree, best_feature, threshold)
        # nodeların oluşturulması
        left_child_node = self._build_tree(left_X, left_y, current_depth+1)
        right_child_node = self._build_tree(right_X, right_y, current_depth+1)
        return Node(feature= best_feature, threshold = threshold, left = left_child_node, right = right_child_node )
    
    def predict(self, X: List[List[float]]): # birden fazla örnek tahmini icin 
        return [self._check_node(x, self.root) for x in X]
    
    def _calculate_gini(self, y: List[int]):
        
        if len(y) ==0:
            return 0
        gini = 1
        length_of_y = len(y)
        distinct_values_of_y = list(set(y))
        for value in distinct_values_of_y:
            number_of_value = y.count(value)
            prob_of_value = number_of_value/length_of_y
            gini -= (prob_of_value ** 2)
        return gini
        
        
    def _most_common_label(self,y): # en çok veride olan sınıfın bulunması
        label_counts = {}
        distinct_labels = list(set(y))
        for label in distinct_labels:
            label_counts[label] = 0  
            for element in y:
                if element == label:
                    label_counts[label] += 1

        max_count = 0               # maksimum geçen verinin sayısı
        most_common_label = None
        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                most_common_label = label
    
        return most_common_label
            
    def _check_node(self, x, node):
        # node kontrolü 
        # node boş işe None döndür
        if node is None:
            return None
        # boş değilse leaf node olup olmadğını kontrol et
        if node.is_leaf_node():
            return node.value
        # boş değilse ve leaf node da değil ise threshold kontrolü yap
        if x[node.feature] <= node.threshold:
            return self._check_node(x, node.left)
        else:   
            return self._check_node(x, node.right)
            
    
        
    
    


