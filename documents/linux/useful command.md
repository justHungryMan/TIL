# Useful Command

1. 파일 일괄 옮기기

```shell script
find ./{directory} -name "_256x256_{name}*" -exec mv {} ./avi{directory} \;
```

2. 파일 이름과 같은 폴더 만들기

```shell script
for file in *.avi
do
  DIR="../data/${file%.*}"
  mkdir "$DIR"
  ffmpeg -i "${file%.*}.avi" "../data/${file%.*}/${file%.*}%05d.jpg"
done

```
3. 현재 위치에서 디렉토리 즉 폴더 개수 세기
```shell script
ls -l | grep ^d | wc -l
```

4. 파일 원하는 갯수 이동
```shell script
ls | head -28440 | xargs -i mv ./"{}" /workspace/data2/
```

