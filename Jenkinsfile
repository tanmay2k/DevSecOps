pipeline {
    agent any

    environment {
        IMAGE_NAME = "expensetracker"
        SONARQUBE_SERVER = "SonarQube" // must match Jenkins config name
        TRIVY_IGNORE_UNFIXED = "true"
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/tanmay2k/DevSecOps/' // replace if needed
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
                    sh "sonar-scanner"
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
            when {
                expression {
                    return true // Set to true if you're pushing to Docker Hub or private registry
                }
            }
            steps {
                sh '''
                docker tag $IMAGE_NAME tlad1/expensetracker:latest
                docker push tlad1/expensetracker:latest
                '''
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh '''
                kubectl apply -f expensetracker-deployment.yaml
                kubectl apply -f expensetracker-service.yaml
                kubectl apply -f postgres-deployment.yaml
                kubectl apply -f postgres-service.yaml
                
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished, regardless of success or you fucking it up.'
        }
    }
}
