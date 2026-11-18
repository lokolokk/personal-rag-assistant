output "instance_id" {
  description = "EC2 instance for the RAG backend."
  value       = aws_instance.server.id
}

output "ecr_repository_url" {
  description = "Where CI pushes the backend image."
  value       = aws_ecr_repository.backend.repository_url
}

output "data_bucket" {
  description = "Private bucket for the data move (transfer/) and backups (backups/)."
  value       = aws_s3_bucket.data.bucket
}

output "server_role_name" {
  description = "Instance role (SSM, ECR pull, /rag/* read)."
  value       = aws_iam_role.server.name
}

output "github_iam_users" {
  description = "IAM users to create access keys for (owner does this in the console)."
  value = var.create_github_users ? [
    aws_iam_user.deploy[0].name,
    aws_iam_user.demo_switch[0].name,
  ] : []
}
