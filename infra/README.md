# infra/ - AWS infrastructure (Terraform)

Creates the cloud home for the backend in `ap-southeast-2`: one EC2 `t3.small`, a private
ECR repo, an SSM role, weekly EBS snapshots and two limited IAM users for GitHub Actions.
State is local and git-ignored. Nothing here contains a secret.

## Run it (PowerShell, from this folder)

```powershell
aws login --profile rag-aws
# Put the temporary credentials into this shell only (nothing is written to disk):
aws configure export-credentials --profile rag-aws --format powershell | Invoke-Expression
$env:AWS_REGION = "ap-southeast-2"

terraform init
terraform fmt -check
terraform validate
terraform plan -out tfplan      # review every line before applying
terraform apply tfplan          # only after the owner approves the plan
```

Teardown: `terraform destroy` (the ECR repo must be empty first).

## What is NOT managed here
- The secrets `/rag/qwen_api_key`, `/rag/qwen_base_url`, `/rag/ngrok_authtoken` (owner-created).
- The AWS Budgets alerts (owner-created in the console).
- GitHub access keys (owner creates them in the console and stores them as GitHub secrets).

## Why IAM users and not GitHub OIDC
The Free-plan account's service control policy denies `iam:*OpenIDConnectProvider*`, so the
keyless OIDC login is not available. Each user can do one job only (see `github_users.tf`).
