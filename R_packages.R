# tidymodels包技术
# R包准备
pkgs_man <- function(package_names){
  # 使用循环操作
  for (pkg in package_names) {
    # 检查包是否已安装
    if (!requireNamespace(pkg, quietly = TRUE)) {
      # 如果包未安装，尝试安装它
      install.packages(pkg)
    }
    
    # 加载包
    library(pkg, character.only = TRUE)
  }
}

# 所需要R包
package_names <- c("tidymodels", "tidyverse")

# 通过函数加载
pkgs_man(package_names = package_names)


