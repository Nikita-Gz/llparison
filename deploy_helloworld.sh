set -e
version=v0.$(date +%s)
echo "Building with version $version"
docker build ./helloworld_container/ -t data6s/helloworld:latest -t data6s/helloworld:$version
docker push data6s/helloworld:latest
kubectl replace --force -f ./helloworld_container/deployment.yaml
