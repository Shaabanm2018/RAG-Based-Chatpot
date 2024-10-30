# Kibana installation and Setup

### To install Kibana with the default docker image version of the current Helm repository

```yaml
helm install kibana elastic/kibana -n elasticsearch
```

wait for sometime and then check the pod (The Kibana pod should be ready and running)

### Alternatively! Installing Kibana with an upgraded version of Docker image

- Create kibana_developement.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kibana
  namespace: elasticsearch
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kibana
  template:
    metadata:
      labels:
        app: kibana
    spec:
      containers:
      - name: kibana
        image: docker.elastic.co/kibana/kibana:8.15.0
        env:
        - name: ELASTICSEARCH_HOSTS
          value: "https://elasticsearch-master:9200"
        - name: ELASTICSEARCH_SERVICEACCOUNTTOKEN
          valueFrom:
            secretKeyRef:
              name: kibana-service-account-token
              key: token
        - name: ELASTICSEARCH_SSL_CERTIFICATEAUTHORITIES
          value: "/usr/share/kibana/config/certs/ca.crt"
        - name: ELASTICSEARCH_SSL_VERIFICATIONMODE
          value: "certificate"
        ports:
        - containerPort: 5601
        volumeMounts:
        - name: elasticsearch-certs
          mountPath: /usr/share/kibana/config/certs # Check logstash installation (Mount Volume for for ca.crt certificate) for this section
          readOnly: true
      volumes:
      - name: elasticsearch-certs
        secret:
          secretName: elasticsearch-master-certs
---
apiVersion: v1
kind: Service
metadata:
  name: kibana
  namespace: elasticsearch
spec:
  selector:
    app: kibana
  ports:
  - port: 5601
    targetPort: 5601
  #  nodePort: 30001  # You can specify a port in the NodePort range
 # type: NodePort (These two line will be uncommented later once we installed nodeport
```

- Create a Kibana service account in Elasticsearch

```yaml
kubectl exec -it elasticsearch-master-0 -n my-namespace-- curl -k -X POST -u elastic:JWp34zQWi7jlTz5Q "https://localhost:9200/_security/service/elastic/kibana/credential/token/kibana_token?
```

If the token already created delete it with 

```yaml
kubectl exec -it elasticsearch-master-0 -n my-namespace --   curl -k -X DELETE -u elastic:6aswkh19Wa0SkLaG   "[https://localhost:9200/_security/service/elastic/kibana/credential/token/kibana_token](https://localhost:9200/_security/service/elastic/kibana/credential/token/kibana_token)"
```

Create a Kubernetes secret for this token

```yaml
kubectl create secret generic kibana-service-account-token --from-literal=token=AAEAAWVsYXN0aWMva2liYW5hL2tpYmFuYV90b2tlbjotV1BjM3A2RlQxZVBDTHdlN08xVmh3 -n elasticsearch
```

- Add this to Kibana_developement.yaml

```yaml
env:
- name: ELASTICSEARCH_SERVICEACCOUNTTOKEN
  valueFrom:
    secretKeyRef:
      name: kibana-service-account-token
      key: token
```

- Apply the file

```yaml
kubectl apply -f kibana-deployment.yaml -n elasticsearch
```

Verify The Kibana Pod Must be running and Ready 

```yaml
kubectl get pod -n elasticsearch -w
```