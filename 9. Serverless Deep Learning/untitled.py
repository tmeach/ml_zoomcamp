ACCOUNT=570673233023
REGION=us-east-1
REGISTRY=clothing-tflite-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazon.com/${REGISTRY}

TAG=clothing-model-xception-v4-001
REGISTRY_URL=${PREFIX}:${TAG}


