# Second private registry, for an image that is built on the owner's laptop and is not part of
# this repository, so CI cannot build it. The owner pushes it by hand.
resource "aws_ecr_repository" "tools" {
  name                 = "${var.name_prefix}-tools"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Only the 5 newest images are kept.
resource "aws_ecr_lifecycle_policy" "tools" {
  repository = aws_ecr_repository.tools.name

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
