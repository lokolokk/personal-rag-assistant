# Private bucket for the server's data: the one-time move from the laptop (transfer/) and
# on-demand backups the laptop can download (backups/). Nothing here is ever public.
resource "aws_s3_bucket" "data" {
  # A random suffix keeps the name unique worldwide without putting the account ID in it.
  bucket_prefix = "${var.name_prefix}-data-"

  # Refuse to delete the bucket while it still holds objects (protects the backups).
  force_destroy = false

  tags = { Name = "${var.name_prefix}-data" }
}

# Block every form of public access, at bucket level.
resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# No ACLs: access is decided only by IAM and the bucket policy.
resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Encrypt every object at rest with S3-managed keys (free, no KMS request charges).
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Old files delete themselves, so private data never piles up and the cost stays at cents.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "transfer-expires"
    status = "Enabled"
    filter {
      prefix = "transfer/"
    }
    expiration {
      days = var.transfer_retention_days
    }
  }

  rule {
    id     = "backups-expire"
    status = "Enabled"
    filter {
      prefix = "backups/"
    }
    expiration {
      days = var.backup_retention_days
    }
  }

  rule {
    id     = "clean-failed-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# Refuse any request that is not over HTTPS.
data "aws_iam_policy_document" "data_bucket" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
    ]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.data_bucket.json

  # Apply after the public-access block so the two never conflict.
  depends_on = [aws_s3_bucket_public_access_block.data]
}
