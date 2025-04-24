pipeline {
    agent any
    environment {
        IMAGE_NAME = "expensetracker"
        SONARQUBE_SERVER = "SonarQube"
        TRIVY_IGNORE_UNFIXED = "true"
        NAMESPACE = "expensetracker"
        MONITORING_NAMESPACE = "monitoring"
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/tanmay2k/DevSecOps.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t ${IMAGE_NAME} ."
                }
            }
        }
        stage('SonarQube Analysis') {
            steps {
                withSonarQubeEnv("${SONARQUBE_SERVER}") {
                    withCredentials([string(credentialsId: 'SONAR_TOKEN', variable: 'SONAR_TOKEN')]) {
                        sh "sonar-scanner -Dsonar.login=${SONAR_TOKEN}"
                    }
                }
            }
        }
        stage('Trivy Scan') {
            steps {
                script {
                    sh '''
                        trivy image --exit-code 1 --severity HIGH,CRITICAL ${IMAGE_NAME} || echo "[!] Vulnerabilities found"
                    '''
                }
            }
        }
        stage('Push to Registry') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
                    script {
                        sh '''
                            echo "$DOCKERHUB_PASS" | docker login -u "$DOCKERHUB_USER" --password-stdin
                            docker tag ${IMAGE_NAME} tlad1/expensetracker:latest
                            docker push tlad1/expensetracker:latest
                        '''
                    }
                }
            }
        }
        stage('Create Namespace') {
            steps {
                script {
                    sh '''
                        kubectl get namespace ${NAMESPACE} || kubectl create namespace ${NAMESPACE}
                    '''
                }
            }
        }
        stage('Deploy to Kind Cluster') {
            steps {
                script {
                    sh '''
                        kubectl apply -f Kubernetes/expensetracker-deployment.yaml -n ${NAMESPACE}
                        kubectl apply -f Kubernetes/expensetracker-service.yaml -n ${NAMESPACE}
                        kubectl apply -f Kubernetes/postgres-deployment.yaml -n ${NAMESPACE}
                        kubectl apply -f Kubernetes/postgres-service.yaml -n ${NAMESPACE}
                        kubectl apply -f Kubernetes/pv.yaml
                        kubectl apply -f Kubernetes/pvc.yaml
                    '''
                }
            }
        }
        stage('Patch Controller Manager Metrics') {
            steps {
                script {
                    sh """
                        echo '[+] Patching kube-controller-manager to expose metrics externally…'
 
                        # find the kind control-plane container (adjust name grep if yours differs)
                        CP=\$(docker ps --format '{{.Names}}' | grep control-plane)
 
                        # sed-replace bind-address in the static pod manifest
                        docker exec \$CP \
                          sed -i 's/--bind-address=127.0.0.1/--bind-address=0.0.0.0/' \
                          /etc/kubernetes/manifests/kube-controller-manager.yaml
 
                        # give kubelet a sec to tear down & spin it back up
                        sleep 10
 
                        # sanity check
                        echo '[+] Testing metrics endpoint…'
                        docker exec \$CP curl -sk https://localhost:10257/metrics | head -n 5
                    """
                }
            }
        }
        stage('Install Monitoring Stack') {
            steps {
                script {
                    sh '''
                        kubectl get namespace ${MONITORING_NAMESPACE} || kubectl create namespace ${MONITORING_NAMESPACE}
                        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
                        helm repo update
                        if ! helm status prometheus -n ${MONITORING_NAMESPACE} > /dev/null 2>&1; then
                            echo "[+] Installing Prometheus stack..."
                            helm install prometheus prometheus-community/kube-prometheus-stack -n ${MONITORING_NAMESPACE}
                        else
                            echo "[!] Prometheus already installed. Skipping install."
                        fi
                    '''
                }
            }
        }
        stage('Port Forward Monitoring') {
            steps {
                script {
                    sh '''
                        echo "[+] Starting port-forwarding for Prometheus and Grafana..."
 
                        # Port-forward Prometheus
                        nohup kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n ${MONITORING_NAMESPACE} 9090:9090 >/dev/null 2>&1 &
                        echo $! > prometheus_pf.pid
 
                        # Port-forward Grafana
                        nohup kubectl port-forward svc/prometheus-grafana -n ${MONITORING_NAMESPACE} 3000:80 >/dev/null 2>&1 &
                        echo $! > grafana_pf.pid
 
                        sleep 5
                    '''
                    echo "[+] Prometheus available at http://localhost:9090"
                    echo "[+] Grafana available at http://localhost:3000"
                }
            }
        }
 
    }
 
    post {
        always {
            echo 'Pipeline finished.'
        }
        failure {
            echo 'Pipeline failed.'
        }
        success {
            echo 'Build, deploy & monitoring setup successful.'
        }
    }
}
