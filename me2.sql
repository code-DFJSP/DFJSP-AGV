/*
 Navicat Premium Dump SQL

 Source Server         : mysql
 Source Server Type    : MySQL
 Source Server Version : 50742 (5.7.42-0ubuntu0.18.04.1)
 Source Host           : 172.23.46.40:3306
 Source Schema         : mysql

 Target Server Type    : MySQL
 Target Server Version : 50742 (5.7.42-0ubuntu0.18.04.1)
 File Encoding         : 65001

 Date: 20/02/2025 23:47:52
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for me2
-- ----------------------------
DROP TABLE IF EXISTS `me2`;
CREATE TABLE `me2`  (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `meid` varchar(255) CHARACTER SET latin1 COLLATE latin1_swedish_ci NULL DEFAULT NULL,
  `emptytime` int(100) NULL DEFAULT NULL,
  `res1` double(255, 6) NULL DEFAULT NULL,
  `res2` double(255, 6) NULL DEFAULT 0.000000,
  PRIMARY KEY (`ID`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 11 CHARACTER SET = latin1 COLLATE = latin1_swedish_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of me2
-- ----------------------------
INSERT INTO `me2` VALUES (1, 'A', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (2, 'B', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (3, 'C', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (4, 'D', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (5, 'E', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (6, 'F', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (7, 'G', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (8, 'H', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (9, 'I', -1, 0.100000, 0.000000);
INSERT INTO `me2` VALUES (10, 'J', -1, 0.100000, 0.000000);

SET FOREIGN_KEY_CHECKS = 1;
