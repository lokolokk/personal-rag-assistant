data "aws_caller_identity" "current" {}

# Free-plan projects ship with a default VPC whose subnets give instances a public IPv4.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Public AWS-owned parameter that always points at the newest Amazon Linux 2023 x86_64 AMI.
data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# The AWS managed key that protects the /rag/* SecureString parameters.
data "aws_kms_alias" "ssm" {
  name = "alias/aws/ssm"
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  subnet_id  = sort(data.aws_subnets.default.ids)[0]
  param_arn  = "arn:aws:ssm:${var.region}:${local.account_id}:parameter/rag"
}
