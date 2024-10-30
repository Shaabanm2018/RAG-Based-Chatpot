# Kubernetes Installation & Setup Guide

## 1. Set up Docker

1. Update package list:
   ```bash
   sudo apt update
   ```

2. Install Docker:
   ```bash
   sudo apt install docker.io -y
   ```

3. Enable Docker on boot:
   ```bash
   sudo systemctl enable docker
   ```

4. Verify Docker status:
   ```bash
   sudo systemctl status docker
   ```

5. Start Docker if not running:
   ```bash
   sudo systemctl start docker
   ```

## 2. Install Kubernetes

1. Add Kubernetes signing key:
   ```bash
   curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
   ```

2. Add Kubernetes repository:
   ```bash
   echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
   sudo apt update
   ```

3. Install Kubernetes tools:
   ```bash
   sudo apt install kubeadm kubelet kubectl
   sudo apt-mark hold kubeadm kubelet kubectl
   ```

4. Verify installation:
   ```bash
   kubeadm version
   ```

## 3. Prepare for Kubernetes Deployment

1. Disable swap:
   ```bash
   sudo swapoff -a
   sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
   ```

2. Load containerd modules:
   ```bash
   echo -e "overlay\nbr_netfilter" | sudo tee /etc/modules-load.d/containerd.conf
   sudo modprobe overlay
   sudo modprobe br_netfilter
   ```

3. Configure Kubernetes networking:
   ```bash
   echo -e "net.bridge.bridge-nf-call-ip6tables = 1\nnet.bridge.bridge-nf-call-iptables = 1\nnet.ip4.ip_forward = 1" | sudo tee /etc/sysctl.d/kubernetes.conf
   sudo sysctl --system
   ```

4. Set unique hostnames:
   - Master node: `sudo hostnamectl set-hostname master-node`
   - Worker nodes: `sudo hostnamectl set-hostname worker01` (repeat for each worker)

5. Edit hosts file:
   ```bash
   sudo nano /etc/hosts
   # Add IP addresses and hostnames of all nodes
   ```

## 4. Initialize Kubernetes on Master Node

1. Configure kubelet:
   ```bash
   echo "KUBELET_EXTRA_ARGS=\"--cgroup-driver=cgroupfs\"" | sudo tee -a /etc/default/kubelet
   sudo systemctl daemon-reload && sudo systemctl restart kubelet
   ```

2. Configure Docker:
   ```json
   {
     "exec-opts": ["native.cgroupdriver=systemd"],
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "100m"
     },
     "storage-driver": "overlay2"
   }
   ```
   Save this to `/etc/docker/daemon.json`

3. Restart Docker:
   ```bash
   sudo systemctl daemon-reload && sudo systemctl restart docker
   ```

4. Configure kubeadm:
   ```bash
   echo "Environment=\"KUBELET_EXTRA_ARGS=--fail-swap-on=false\"" | sudo tee -a /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
   sudo systemctl daemon-reload && sudo systemctl restart kubelet
   ```

5. Initialize cluster:
   ```bash
   sudo kubeadm init --control-plane-endpoint=master-node --upload-certs
   ```

6. Set up kubeconfig:
   ```bash
   mkdir -p $HOME/.kube
   sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
   sudo chown $(id -u):$(id -g) $HOME/.kube/config
   ```

## 5. Deploy Pod Network

1. Install Flannel:
   ```bash
   kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
   ```

2. Untaint master node:
   ```bash
   kubectl taint nodes --all node-role.kubernetes.io/control-plane-
   ```

## 6. Join Worker Nodes to Cluster

1. On worker nodes:
   ```bash
   sudo systemctl stop apparmor && sudo systemctl disable apparmor
   sudo systemctl restart containerd.service
   sudo kubeadm join [master-node-ip]:6443 --token [token] --discovery-token-ca-cert-hash sha256:[hash]
   ```

2. Verify cluster status (on master node):
   ```bash
   kubectl get nodes
   ```

## Troubleshooting

- If you face network plugin issues, you may need to reinitialize the cluster with the network flannel plugin or the IP address of the node.

