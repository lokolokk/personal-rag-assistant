# GitHub Actions needs AWS access for (a) deploying a new image and (b) the Start/Stop demo buttons.
# The preferred way (OIDC, no stored keys) is blocked on this Free-plan project: the account's
# service control policy denies iam:*OpenIDConnectProvider*. So two separate least-privilege IAM
# users are used instead. Terraform creates ONLY the users and their policies. The owner creates
# each access key in the console and stores it as a GitHub secret, so no key ever enters Terraform
# state or this repository.

locals {
  instance_arn = aws_instance.server.arn
}

# ---------- deploy user: push an image, tell the server to run it ----------

resource "aws_iam_user" "deploy" {
  count = var.create_github_users ? 1 : 0
  name  = "${var.name_prefix}-gha-deploy"
}

data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPushOneRepo"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.backend.arn]
  }

  statement {
    sid       = "SetDesiredTag"
    actions   = ["ssm:PutParameter", "ssm:GetParameter"]
    resources = [aws_ssm_parameter.desired_image_tag.arn]
  }

  statement {
    sid     = "RunDeployCommand"
    actions = ["ssm:SendCommand"]
    resources = [
      local.instance_arn,
      "arn:aws:ssm:${var.region}::document/AWS-RunShellScript",
    ]
  }

  # These two cannot be limited to one resource.
  statement {
    sid       = "ReadCommandResult"
    actions   = ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"]
    resources = ["*"]
  }
}

resource "aws_iam_user_policy" "deploy" {
  count  = var.create_github_users ? 1 : 0
  name   = "deploy"
  user   = aws_iam_user.deploy[0].name
  policy = data.aws_iam_policy_document.deploy.json
}

# ---------- demo-switch user: start / stop the one server, nothing else ----------

resource "aws_iam_user" "demo_switch" {
  count = var.create_github_users ? 1 : 0
  name  = "${var.name_prefix}-gha-demo-switch"
}

data "aws_iam_policy_document" "demo_switch" {
  statement {
    sid       = "StartStopOneInstance"
    actions   = ["ec2:StartInstances", "ec2:StopInstances"]
    resources = [local.instance_arn]
  }

  # Describe calls cannot be limited to one resource.
  statement {
    sid       = "ReadInstanceState"
    actions   = ["ec2:DescribeInstances", "ec2:DescribeInstanceStatus"]
    resources = ["*"]
  }
}

resource "aws_iam_user_policy" "demo_switch" {
  count  = var.create_github_users ? 1 : 0
  name   = "demo-switch"
  user   = aws_iam_user.demo_switch[0].name
  policy = data.aws_iam_policy_document.demo_switch.json
}
