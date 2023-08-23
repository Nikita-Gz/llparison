set -e
version=v0.$(date +%s)
echo "Building frontend with version $version"
docker build ./llpfrontend/ -t data6s/llpfrontend:latest -t data6s/llpfrontend:$version
docker push data6s/llpfrontend:latest
kubectl replace --force -f ./llpfrontend/django_svc.yaml
kubectl replace --force -f ./llpfrontend/django_deployment.yaml
