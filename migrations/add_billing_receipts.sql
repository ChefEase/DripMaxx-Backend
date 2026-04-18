create table if not exists billing_receipts (
  id varchar(36) primary key,
  user_id varchar(36) not null references users(id) on delete cascade,
  platform text not null,
  product_id text not null,
  purchase_token text,
  transaction_id text,
  raw_receipt json not null default '{}',
  verified_at timestamptz not null default now(),
  expires_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists idx_billing_receipts_user_id
  on billing_receipts (user_id);

create unique index if not exists uq_billing_receipts_purchase_token
  on billing_receipts (platform, purchase_token)
  where purchase_token is not null;

create unique index if not exists uq_billing_receipts_transaction_id
  on billing_receipts (platform, transaction_id)
  where transaction_id is not null;
