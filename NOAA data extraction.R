library(xml2)
library(rvest)
library(rNOMADS)
#Latitude et longitude de Île de France
lat <- 48.8499
lon <- 2.6370

abbrev <- "gfsanl"
model.seq <- seq(as.Date(as.character(20140101), format = "%Y%m%d"), by = "day", length.out = 730) 
model.seq.format <- format(model.seq, format = "%Y%m%d")
model.run <- c(00, 06, 12, 18)
pred <- 00
#nombre de jours
d <- 1
#nombre de prédictions
p <- 4
Matrix_tmp <- matrix(data = NA, nrow = d, ncol = p)
Matrix_gust <- matrix(data = NA, nrow = d, ncol = p)
Matrix_pwat <- matrix(data = NA, nrow = d, ncol = p)

ptm <- proc.time()
for(i in 1:d)
{
  for(k in 1:p)
  {
    ## Not run:
    ##télécharger des données
    model.info <- ArchiveGribGrab(abbrev, model.seq.format[i], model.run[k], pred, file.type = "grib2")
    #TMP: température disponible quand LEVEL = 2 m above ground
    #GUST: vitesse du vent surface
    #PWAT: precipitable water for the entire atmosphere/millimeters or inches
    variables <- c("TMP", "GUST", "PWAT")
    levels     <- c("2 m above ground", "entire atmosphere \\(considered as a single layer\\)", "surface")
    domain   <- c(lon - 1, lon + 1, lat + 1, lat - 1) 
    ##lire des données
    model.data <- ReadGrib(model.info$file.name, levels, variables, domain = domain)
    #prendre des données de certaine point
    profile <- BuildProfile(model.data, lon, lat, TRUE)
    #écrire des données de chaque 6 heures à une liste
    tmp <- profile[[1]]$profile.data[2,2,1] - 272.15
    gust <- profile[[1]]$profile.data[1,1,1]
    pwat <- profile[[1]]$profile.data[3,3,1]
    Matrix_tmp[i,k] <- tmp
    Matrix_gust[i,k] <- gust
    Matrix_pwat[i,k] <- pwat
    
  }
}
data_tmp <- write.table(Matrix_tmp, file = "/data/users/ling.jin/data_tmp.csv", row.names = T, col.names = c("00","06","12","18"),sep = ";")
data_gust <- write.table(Matrix_gust, file = "/data/users/ling.jin/data_gust.csv", row.names = T, col.names = c("00","06","12","18"),sep = ";")
data_pwat <- write.table(Matrix_pwat, file = "/data/users/ling.jin/data_pwat.csv", row.names = T, col.names = c("00","06","12","18"),sep = ";")

proc.time() - ptm

model.info <- ArchiveGribGrab(abbrev, 20140101, 00, 0, file.type = "grib2")

