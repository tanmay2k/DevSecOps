ipipeline {
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
                        # Prometheus
                        nohup kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n ${MONITORING_NAMESPACE} 9090:9090 >/dev/null 2>&1 &

                        # Grafana
                        nohup kubectl port-forward svc/prometheus-grafana -n ${MONITORING_NAMESPACE} 3000:80 >/dev/null 2>&1 &
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

