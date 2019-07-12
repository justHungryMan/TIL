# Docker with deep learning

### Docker Settings

[이제는, 딥러닝 개발환경도Docker로 올려보자!!](http://moducon.kr/2018/wp-content/uploads/sites/2/2018/12/leesangsoo_slide.pdf)

---

### Docker Data 관리

##### Volume

- `volume`은 Docker가 관리하는 Host의 File System 일부에 Data 가 저장된다.
- Non-Docker 프로세스는 File System의 해당 부분을 수정해서는 안된다.
- Docker에서 Data를 존속시킬 수 있는 Best한 방법


#### Mount

- `bind mount`는 Data가 Host System의 어디에든지 저장 가능하다.
- Docker Host 또는 Docker Container.의 Non-Docker 프로세서들이 언제든지 저장된 Data를 수정 가능하다


#### Command line
```
docker run -v {local path}:{container path} pytorch/pytorch:{}
```

---

