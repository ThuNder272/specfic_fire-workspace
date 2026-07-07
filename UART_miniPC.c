#include "UART_miniPC.h"

uint8_t usart1_buf[2][UART_RX_BUF_LENGHT];
uint8_t usart1_send_buf[2][UART_TX_BUF_LENGHT];
data_miniPC_aim_t data_miniPC_aim = {0};
data_miniPC_aim_send_t data_miniPC_aim_send = {0};

/* ========== 卡尔曼滤波实例 ========== */
KalmanFilter_t kf_yaw;   // yaw轴卡尔曼滤波器
KalmanFilter_t kf_pitch; // pitch轴卡尔曼滤波器

extern UART_HandleTypeDef huart1;
extern DMA_HandleTypeDef hdma_usart1_rx;
extern DMA_HandleTypeDef hdma_usart1_tx;

extern uint8_t usart6_buf[2][UART_RX_BUF_LENGHT];
extern UART_HandleTypeDef huart6;
extern DMA_HandleTypeDef hdma_usart6_rx;
extern DMA_HandleTypeDef hdma_usart6_tx;

uint8_t tx[9] = {0};
//uint8_t nav_mode = 0;
uint8_t aim_mode = 0;

void miniPC_aim_Init(void)
{
    uart_miniPC_aim_Init(usart1_buf[0], usart1_buf[1], UART_RX_BUF_LENGHT);

    // 防止未初始化直接Update导致 0/0->NaN，进而让aim_yaw/aim_pitch变成0（表现为“检测到目标就停住”）
    Kalman_Init(&kf_yaw, KALMAN_Q_DEFAULT, KALMAN_R_DEFAULT, 0.0f);
    Kalman_Init(&kf_pitch, KALMAN_Q_DEFAULT, KALMAN_R_DEFAULT, 0.0f);
    // uart_miniPC_send_Init(usart1_send_buf[0], usart1_send_buf[1], UART_TX_BUF_LENGHT);
    //fifo_s_init(&uart_miniPC_fifo, uart_miniPC_fifo_buf, UART_MINIPC_FIFO_BUF_LENGTH);
}

void uart_miniPC_aim_Init(uint8_t *rx1_buf, uint8_t *rx2_buf, uint16_t dma_buf_num)
{
    // enable the DMA transfer for the receiver and tramsmit request
    // 使能DMA串口接收和发送
    SET_BIT(huart1.Instance->CR3, USART_CR3_DMAR);
    SET_BIT(huart1.Instance->CR3, USART_CR3_DMAT);
    // enalbe idle interrupt
    // 使能空闲中断
    __HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE); // 开启中断
    // disable DMA
    // 失效DMA
    __HAL_DMA_DISABLE(&hdma_usart1_rx);

    while (hdma_usart1_rx.Instance->CR & DMA_SxCR_EN) // 使能双缓冲区
    {
        __HAL_DMA_DISABLE(&hdma_usart1_rx);
    }

    __HAL_DMA_CLEAR_FLAG(&hdma_usart1_rx, DMA_LISR_TCIF1);

    hdma_usart1_rx.Instance->PAR = (uint32_t)&(USART1->DR);
    // memory buffer 1
    // 内存缓冲区1
    hdma_usart1_rx.Instance->M0AR = (uint32_t)(rx1_buf); // 设备初始缓冲区，数据存到指定内存区域，地址换为32位指针
    // memory buffer 2
    // 内存缓冲区2
    hdma_usart1_rx.Instance->M1AR = (uint32_t)(rx2_buf); // 设备备用缓冲区
    // data length
    // 数据长度
    __HAL_DMA_SET_COUNTER(&hdma_usart1_rx, dma_buf_num);

    // enable double memory buffer
    // 使能双缓冲区
    SET_BIT(hdma_usart1_rx.Instance->CR, DMA_SxCR_DBM); // 启用双缓冲模式

    // enable DMA
    // 使能DMA
    __HAL_DMA_ENABLE(&hdma_usart1_rx);

    // disable DMA
    // 失效DMA
    __HAL_DMA_DISABLE(&hdma_usart1_tx);

    while (hdma_usart1_tx.Instance->CR & DMA_SxCR_EN)
    {
        __HAL_DMA_DISABLE(&hdma_usart1_tx);
    }

    hdma_usart1_tx.Instance->PAR = (uint32_t)&(USART1->DR); // DR数据寄存器，外设地址
}

//void uart_miniPC_nav_read_data(uint8_t *frame)
//{
//    memcpy(&data_miniPC_nav, frame, sizeof(data_miniPC_nav_t));
//}

void uart_miniPC_aim_read_data(uint8_t *frame)
{
    static uint8_t last_aim_ready = 0;

    memcpy(&data_miniPC_aim, frame, sizeof(data_miniPC_aim_t));

    while (data_miniPC_aim.aim_yaw > 18000)
    {
        data_miniPC_aim.aim_yaw -= 36000;
    }
    while (data_miniPC_aim.aim_yaw < -18000)
    {
        data_miniPC_aim.aim_yaw += 36000;
    }

    while (data_miniPC_aim.aim_pitch > 18000)
    {
        data_miniPC_aim.aim_pitch -= 36000;
    }
    while (data_miniPC_aim.aim_pitch < -18000)
    {
        data_miniPC_aim.aim_pitch += 36000;
    }

    data_miniPC_aim.aim_yaw = data_miniPC_aim.aim_yaw / 100;
    data_miniPC_aim.aim_pitch = data_miniPC_aim.aim_pitch / 100;

    // 目标从“未检测”->“检测到”时，用当前测量值重置滤波器，避免第一次锁定因为历史状态导致输出异常/滞后
    if (data_miniPC_aim.aim_ready == 1 && last_aim_ready == 0)
    {
        Kalman_Init(&kf_yaw, kf_yaw.Q, kf_yaw.R, (float)data_miniPC_aim.aim_yaw);
        Kalman_Init(&kf_pitch, kf_pitch.Q, kf_pitch.R, (float)data_miniPC_aim.aim_pitch);
    }

    float filtered_yaw = (data_miniPC_aim.aim_ready == 1) ? Kalman_Update(&kf_yaw, (float)data_miniPC_aim.aim_yaw)
                                                          : (float)data_miniPC_aim.aim_yaw;
    float filtered_pitch = (data_miniPC_aim.aim_ready == 1) ? Kalman_Update(&kf_pitch, (float)data_miniPC_aim.aim_pitch)
                                                            : (float)data_miniPC_aim.aim_pitch;

    data_miniPC_aim.aim_yaw = (int16_t)filtered_yaw;
    data_miniPC_aim.aim_pitch = (int16_t)filtered_pitch;

    last_aim_ready = data_miniPC_aim.aim_ready;
}

void uart_miniPC_aim_unpack_data(int n)
{
    const int frame_len = (int)sizeof(data_miniPC_aim_t);
    for (int i = 0; i <= (int)UART_RX_BUF_LENGHT - frame_len; i++)
    {
        if (usart1_buf[n][i] == AIM_HEADER_SOF && usart1_buf[n][i + frame_len - 1] == AIM_HEADER_EOF)
        {
            // 必须用当前缓冲区 n，并从帧头位置 i 开始，否则会读到旧帧/错误帧，表现为“锁定后数据不更新、炮口停住”
            uart_miniPC_aim_read_data(&usart1_buf[n][i]);
            aim_mode = 1;
        }
    }
}

void uart_miniPC_aim_send_data(void)
{
    if (aim_mode == 1)
    {
        HAL_UART_Transmit_DMA(&huart1, &data_miniPC_aim_send, 8);// 首次调用dma发送，后续因为会触发传输完成中断，因而无需再次调用
        aim_mode = 0;
    }
}

void USART1_IRQHandler(void)
{
    static volatile uint8_t res;
    HAL_UART_IRQHandler(&huart1);
    if (USART1->SR & UART_FLAG_IDLE)
    {
        __HAL_UART_CLEAR_PEFLAG(&huart1);

        static uint16_t this_time_rx_len = 0;

        if ((huart1.hdmarx->Instance->CR & DMA_SxCR_CT) == RESET) // 如果是缓冲区0
        {
            __HAL_DMA_DISABLE(huart1.hdmarx);                                             // 失能dma
            __HAL_DMA_SET_COUNTER(huart1.hdmarx, UART_RX_BUF_LENGHT);                     // 原本存储空间
            huart1.hdmarx->Instance->CR |= DMA_SxCR_CT;                                   // 将缓冲区变成1
            __HAL_DMA_ENABLE(huart1.hdmarx);                                              // 使能dma
            uart_miniPC_aim_unpack_data(0);
        }

        else // 如果是缓冲区1
        {
            __HAL_DMA_DISABLE(huart1.hdmarx);                                             // 失能dma
            __HAL_DMA_SET_COUNTER(huart1.hdmarx, UART_RX_BUF_LENGHT);                     // 原本存储空间
            huart1.hdmarx->Instance->CR &= ~(DMA_SxCR_CT);                                // 将缓冲区变成0
            __HAL_DMA_ENABLE(huart1.hdmarx);                                              // 使能dma
            uart_miniPC_aim_unpack_data(1);      
        }
        if (this_time_rx_len > 0)
        {
            uart_miniPC_aim_send_data();
        }
    }
}

int float_2_uint(float x, float x_min, float x_max, int bits)
{
    /* Converts a float to an unsigned int, given range and number of bits */
    float span = x_max - x_min;
    float offset = x_min;
    return (int)((x - offset) * ((float)((1 << bits) - 1)) / span);
}

/* ========== 卡尔曼滤波实例 ========== */
    /**
     * @brief  卡尔曼滤波器初始化
     * @param  kf: 卡尔曼滤波器指针
     * @param  Q: 过程噪声协方差（0.001~0.1，越大跟踪越快但越不平滑）
     * @param  R: 测量噪声协方差（0.1~10，越大越信任预测值）
     * @param  init_val: 初始值
     */
void Kalman_Init(KalmanFilter_t *kf, float Q, float R, float init_val)
{
    // 兜底：避免Q/R被传成0导致后续 0/0
    kf->Q = (Q > 0.0f) ? Q : KALMAN_Q_DEFAULT;
    kf->R = (R > 0.0f) ? R : KALMAN_R_DEFAULT;
    kf->P = 1.0f;      // 初始估计误差协方差
    kf->K = 0.0f;      // 初始卡尔曼增益
    kf->x = init_val;  // 初始状态估计
    kf->inited = 1;
}
    
    /**
     * @brief  卡尔曼滤波更新（每次接收到新测量值时调用）
     * @param  kf: 卡尔曼滤波器指针
     * @param  measurement: 新的测量值（如 aim_yaw 或 aim_pitch）
     * @return 滤波后的最优估计值
     */
float Kalman_Update(KalmanFilter_t *kf, float measurement)
{
    if (kf->inited == 0)
    {
        Kalman_Init(kf, KALMAN_Q_DEFAULT, KALMAN_R_DEFAULT, measurement);
        return kf->x;
    }

    if (kf->Q <= 0.0f)
    {
        kf->Q = KALMAN_Q_DEFAULT;
    }
    if (kf->R <= 0.0f)
    {
        kf->R = KALMAN_R_DEFAULT;
    }

    // 预测步骤：P_predict = P_last + Q
    kf->P = kf->P + kf->Q;

    // 更新步骤
    const float denom = kf->P + kf->R;
    if (denom <= 1e-9f)
    {
        // 极端情况下避免除0：直接跟随测量并重置协方差
        kf->x = measurement;
        kf->P = 1.0f;
        kf->K = 0.0f;
        return kf->x;
    }

    // K = P_predict / (P_predict + R)
    kf->K = kf->P / denom;

    // x_optimal = x_predict + K * (measurement - x_predict)
    kf->x = kf->x + kf->K * (measurement - kf->x);

    // P_optimal = (1 - K) * P_predict
    kf->P = (1.0f - kf->K) * kf->P;

    return kf->x;
}
