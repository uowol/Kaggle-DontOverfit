from pydantic import BaseModel, create_model


## class PredIn ## 
# 동적으로 클래스 속성 정의하기
fields = {f"f{i}": (float, 0.0) for i in range(300)}    # 각 필드는 float 타입이고, 기본값으로 0.0을 사용합니다.
# [ X ] -> PredIn = type("PredIn", (BaseModel,), fields)
PredIn = create_model("PredIn", **fields)               # PredIn 클래스를 생성합니다. fields에 메소드를 정의할 수도 있습니다.

## class PredOut ## 
class PredOut(BaseModel):
    target: int