pipeline {
    agent any

    environment {
        IMAGE_NAME = "expensetracker"
        SONARQUBE_SERVER = "SonarQube" 
        TRIVY_IGNORE_UNFIXED = "true"
        NAMESPACE = "expensetracker"
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
                    sh "docker build -t $IMAGE_NAME ."
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
                sh '''
                trivy image --exit-code 1 --severity HIGH,CRITICAL ${IMAGE_NAME} || echo "[!] Vulnerabilities found"
                '''
            }
        }

        stage('Push to Registry') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    retry(3) {
                        sh '''
                        docker tag $IMAGE_NAME tlad1/expensetracker:latest
                        docker push tlad1/expensetracker:latest
                        '''
                    }
                }
            }
        }

        stage('Create Namespace') {
            steps {
                sh '''
                kubectl get namespace $NAMESPACE || kubectl create namespace $NAMESPACE
                '''
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh '''
                kubectl apply -f Kubernetes/expensetracker-deployment.yaml -n $NAMESPACE
                kubectl apply -f Kubernetes/expensetracker-service.yaml -n $NAMESPACE
                kubectl apply -f Kubernetes/postgres-deployment.yaml -n $NAMESPACE
                kubectl apply -f Kubernetes/postgres-service.yaml -n $NAMESPACE
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
    }
}
