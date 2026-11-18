# Private registry for the owner-only admin tool. Its code is not in this repository, so
# CI cannot build it: the owner builds and pushes it from the laptop (see docs, P5.2).
resource "aws_ecr_repository" "admin" {
  name                 = "${var.name_prefix}-admin"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Only the 5 newest images are kept.
resource "aws_ecr_lifecycle_policy" "admin" {
  repository = aws_ecr_repository.admin.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 5 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}
