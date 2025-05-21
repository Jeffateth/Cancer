## -----------------------------
## 0. INSTALL PACKAGES (Run once)
## -----------------------------
install.packages("devtools", dependencies = TRUE)
devtools::install_github("BIMSBbioinfo/VoltRon")

install.packages("rJava")
install.packages("BiocManager")
install.packages("R.utils")

BiocManager::install("rhdf5")
BiocManager::install("RBioFormats")

## -----------------------------
## 1. LOAD LIBRARIES
## -----------------------------
library(VoltRon)
library(rJava)
library(R.utils)
library(BiocManager)

# Test Java works
.jinit()
.jcall("java/lang/System", "S", "getProperty", "java.version")

## -----------------------------
## 2. DEFINE FILE PATHS
## -----------------------------
base_path <- "/Users/jianzhouyao/Cancer/Pancreatic Cancer with Xenium Human Multi-Tissue and Cancer Panel/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs"
he_image_path <- "/Users/jianzhouyao/Cancer/Pancreatic Cancer with Xenium Human Multi-Tissue and Cancer Panel/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif"

## -----------------------------
## 3. IMPORT XENIUM DATA
## -----------------------------
Xenium <- importXenium(base_path, sample_name = "XeniumPancreas")

## -----------------------------
## 4. SAFELY INSPECT H&E IMAGE CHANNELS
## -----------------------------
# Step 1: Load without forcing channels
HnE_test <- importImageData(he_image_path, sample_name = "TestHnE", channel_names = "H&E")

# Step 2: Print available channels
vrImageChannelNames(HnE_test)

## -----------------------------
## 6. LAUNCH SHINY APP FOR ALIGNMENT
## -----------------------------
# Optional: Force shiny to open in browser
options(shiny.launch.browser = TRUE)

# Register images
aligned <- registerSpatialData(object_list = list(Xenium, HnE_test))

## -----------------------------
## 7. SAVE ALIGNMENT PARAMETERS
## -----------------------------
mapping_parameters <- aligned$mapping_parameters
transformation_matrix <- mapping_parameters[[1]]$matrix

write.csv(transformation_matrix,
          file = file.path(base_path, "alignment_transformation_matrix.csv"),
          row.names = FALSE)

saveRDS(mapping_parameters, 
        file = file.path(base_path, "mapping_parameters.rds"))

## -----------------------------
## 8. (OPTIONAL) ADD H&E CHANNEL TO XENIUM OBJECT
## -----------------------------
HnE_registered <- aligned$registered_spat[[2]]
vrImages(Xenium[["Assay1"]], channel = "H&E") <- vrImages(HnE_registered, name = "main_reg", channel = "H&E")

vrImageChannelNames(Xenium)

## -----------------------------
## 9. (OPTIONAL) VISUALIZE
## -----------------------------
vrImages(Xenium, channel = "H&E", scale.perc = 5)
