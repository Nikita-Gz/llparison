set -e
version=v0.$(date +%s)
echo "Building with version $version"
docker build ./model_tracking/ -t data6s/model_tracking:latest -t data6s/model_tracking:$version
docker push data6s/model_tracking:latest
kubectl replace --force -f ./model_tracking/deployment.yaml
