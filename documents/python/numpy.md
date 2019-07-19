# Numpy

### 리스트에서 i번째 제외하고 합치기 (오른쪽에)

```python
np.vstack(X_train_folds[:i] + X_train_folds[i + 1 :])
```

### 리스트에서 i번째 제외하고 합치기 (아래쪽에)

```python
np.hstack(Y_train_folds[:i] + Y_train_folds[i + 1:])
```

### 열 성분 더하기
```python
np.sum(A, axis = 1)
```

### Compute distances no loops
```python
dists += np.sum(X ** 2, axis =1).reshape(num_test, 1)
dists += np.sum(self.X_train **2, axis = 1).reshape(num_train, 1).T
dists -= 2 * X.dot(self.X_train.T)
dists = np.sqrt(dists)
```
수학적으로 열성분이 어디에 더해지고 행성분이 어디에 더해지는지 행렬로 그려보면 위와같의 표현 가능하다.

### item을 기준으로 index를 sorting
```python
sortedArgs = np.argsort(dists[i, :])
index = sortedArgs[:k]
closest_y = self.y_train[index]
```

### list 안에서 가장 많이 나온 item 찾기
```python
counts = np.bincount(closest_y)
y_pred[i] = np.argmax(counts)
```