# Weekly encrypted snapshot of the server volume, keep the newest N.
resource "aws_dlm_lifecycle_policy" "weekly" {
  description        = "Weekly snapshot of the RAG server volume"
  execution_role_arn = aws_iam_role.dlm.arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]
    target_tags    = { Snapshot = "weekly" }

    schedule {
      name = "weekly"

      create_rule {
        # Sunday 17:00 UTC.
        cron_expression = "cron(0 17 ? * SUN *)"
      }

      retain_rule {
        count = var.snapshot_retention
      }

      copy_tags = true
    }
  }
}
