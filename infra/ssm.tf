# The image tag the server should run. CD overwrites it after each successful build; the value
# here is only a placeholder. The three secret parameters (/rag/qwen_api_key, /rag/qwen_base_url,
# /rag/ngrok_authtoken) are created by the owner in the console and are deliberately NOT managed
# here, so their values never enter Terraform state.
resource "aws_ssm_parameter" "desired_image_tag" {
  name  = "/rag/desired_image_tag"
  type  = "String"
  value = "none"

  lifecycle {
    ignore_changes = [value]
  }
}

# Non-secret settings read by the server at boot (see deploy/aws/rag-up.sh).
resource "aws_ssm_parameter" "ngrok_domain" {
  name  = "/rag/ngrok_domain"
  type  = "String"
  value = var.ngrok_domain
}

resource "aws_ssm_parameter" "frontend_origin" {
  name  = "/rag/frontend_origin"
  type  = "String"
  value = var.frontend_origin
}

# Name of the private data bucket, so the backup scripts never hard-code it.
resource "aws_ssm_parameter" "data_bucket" {
  name  = "/rag/data_bucket"
  type  = "String"
  value = aws_s3_bucket.data.bucket
}

# Master switch for paid Qwen calls. Starts "false" so the first tests spend nothing.
# Flip it with: aws ssm put-parameter --name /rag/provider_enabled --value true --overwrite
resource "aws_ssm_parameter" "provider_enabled" {
  name  = "/rag/provider_enabled"
  type  = "String"
  value = "false"

  lifecycle {
    ignore_changes = [value]
  }
}
