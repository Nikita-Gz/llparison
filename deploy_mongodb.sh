kubectl apply -f ./mongodb/mongo_setup.yaml
helm repo add bitnami https://charts.bitnami.com/bitnami
helm upgrade --install mongodb bitnami/mongodb --set volumePermissions.enabled=true --set auth.rootPassword="root"
