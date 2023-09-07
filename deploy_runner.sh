set -e
version=v0.$(date +%s)
echo "Building with version $version"
docker build ./model_running/ -t data6s/model_running:latest -t data6s/model_running:$version
docker push data6s/model_running:latest
kubectl replace --force -f ./secrets.yaml
kubectl replace --force -f ./model_running/deployment.yaml
