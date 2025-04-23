pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds')
    }

    stages {
        stage('Sanity Check') {
            steps {
                echo 'Pipeline’s alive, bitch.'
            }
        }

        stage('Clone Repo') {
            steps {
                git 'https://github.com/tanmay2k/DevSecOps'
            }
        }

        stage('Docker Build') {
            steps {
                script {
                    def imageName = "tlad1/expensetracker-web:latest"
                    sh "docker build -t ${imageName} ."
                }
            }
        }

        stage('Docker Push') {
            steps {
                script {
                    def imageName = "tlad1/expensetracker-web:latest"
                    sh """
                        echo ${DOCKERHUB_CREDENTIALS_PSW} | docker login -u ${DOCKERHUB_CREDENTIALS_USR} --password-stdin
                        docker push ${imageName}
                    """
                }
            }
        }

        stage('K8s Dry Deploy') {
            steps {
                sh 'kubectl version --client'
                sh 'kubectl get nodes'
            }
        }
    }

    post {
        success {
            echo 'Pipeline ran smoother than your last relationship, ya clown.'
        }
        failure {
            echo 'Pipeline crashed like a drunk on a moped. Fix it, ya donut.'
        }
    }
}

