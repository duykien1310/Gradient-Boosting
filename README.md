# Gradient-Boosting
Gradient Boosting


Hàm load_data:
Tải dữ liệu từ các file CSV.
Kiểm tra sự tồn tại của các file và thông báo lỗi nếu không tìm thấy.
Hàm preprocess_data:
Xử lý Thiếu Giá Trị:
Đối với các cột số, sử dụng trung bình để điền giá trị thiếu.
Đối với các cột phân loại, sử dụng giá trị xuất hiện nhiều nhất để điền.
Mã Hóa Các Biến Phân Loại:
Sử dụng LabelEncoder để chuyển đổi các biến phân loại thành số.
Xử lý các giá trị chưa thấy trong tập kiểm tra bằng cách thêm nhãn 'Unknown'.
Chuẩn Hóa Dữ Liệu:
Sử dụng StandardScaler để chuẩn hóa các cột số.
Hàm train_model:
Huấn luyện mô hình GradientBoostingClassifier với các tham số mặc định.
Bạn có thể điều chỉnh các tham số này tùy theo yêu cầu.
Hàm evaluate_model:
Tính toán ma trận confusion matrix, recall, precision, F1-score.
In ra báo cáo phân loại đầy đủ.
Hàm plot_feature_importances:
Vẽ biểu đồ tầm quan trọng của các đặc trưng dựa trên mô hình Gradient Boosting.
Giúp hiểu rõ hơn về các đặc trưng đóng góp nhiều nhất vào mô hình.
Hàm hyperparameter_tuning (Tùy chọn):
Sử dụng GridSearchCV để tìm các siêu tham số tốt nhất cho mô hình.
Tham số được tìm kiếm bao gồm n_estimators, learning_rate, max_depth, và subsample.
Hàm main:
Định nghĩa các đường dẫn đến dữ liệu.
Tải và tiền xử lý dữ liệu.
Nếu tập kiểm tra không có nhãn, chia tập huấn luyện thành tập huấn luyện và tập kiểm tra.
Huấn luyện mô hình và đánh giá.
Vẽ biểu đồ tầm quan trọng các đặc trưng.
Các bước tùy chọn như tối ưu hóa siêu tham số, lưu mô hình và lưu kết quả đánh giá.
Hướng Dẫn Sử Dụng
Chuẩn Bị Môi Trường:
Đảm bảo rằng bạn đã cài đặt các thư viện cần thiết. Bạn có thể cài đặt chúng bằng lệnh:
bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn


Cấu Trúc Thư Mục:
Đảm bảo rằng thư mục dự án của bạn có cấu trúc như sau:
kotlin
Copy code
project/
├── data/
│   ├── train.csv
│   └── test.csv
└── loan_approval_classifier.py


Chạy Script:
Mở terminal, điều hướng đến thư mục dự án và chạy lệnh:
bash
Copy code
python loan_approval_classifier.py


Tùy Chỉnh:
Tên Cột Nhãn (TARGET_COL): Thay 'target' bằng tên cột nhãn thực tế trong dữ liệu của bạn.
Hyperparameter Tuning: Nếu bạn muốn thực hiện tối ưu hóa siêu tham số, hãy bỏ dấu """ ở phần tương ứng trong hàm main.
Lưu Mô Hình và Kết Quả: Bạn có thể kích hoạt các phần lưu mô hình và kết quả bằng cách bỏ dấu """ trong hàm main.
Lưu Ý Quan Trọng
Kiểm Tra Định Dạng Dữ Liệu: Đảm bảo rằng các cột trong train.csv và test.csv phù hợp với các bước tiền xử lý trong mã. Nếu có sự khác biệt, bạn cần điều chỉnh mã tương ứng.
Xử Lý Dữ Liệu Không Cân Bằng: Nếu tập dữ liệu của bạn có sự mất cân bằng giữa các lớp, bạn có thể cần áp dụng các kỹ thuật như SMOTE, RandomOverSampler, hoặc điều chỉnh trọng số lớp trong mô hình.
Giám Sát Hiệu Năng: Luôn theo dõi các chỉ số đánh giá để đảm bảo mô hình không bị overfitting hoặc underfitting.