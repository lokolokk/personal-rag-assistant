terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# Credentials come from the environment (temporary values exported from `aws login`),
# never from this file. See infra/README.md.
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "personal-rag-assistant"
      ManagedBy = "terraform"
    }
  }
}
